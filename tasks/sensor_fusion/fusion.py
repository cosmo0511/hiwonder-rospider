"""뎁스 카메라 + IMU 융합 판정 로직.

이 모듈은 **Kit(omni) 에 의존하지 않는다.** 들어오는 것은 텐서뿐이라
`AppLauncher` 없이 import 하고 단위 테스트할 수 있다. Isaac Sim 쪽 코드
(`run_fusion_demo.py`) 는 센서에서 읽은 텐서를 여기 넘기기만 한다.

융합의 요지
-----------
뎁스 카메라 하나만으로 "앞에 장애물이 있나" 를 판단하면, 로봇이 기울어진 순간
**바닥이 장애물로 보인다.** 아래로 14° 기울어 설치된 카메라가 몸체 pitch 로 추가로
15° 숙여지면 시선이 바닥에 꽂히고, 뎁스 최솟값은 장애물이 아니라 바닥 거리가 된다.

IMU 는 거리를 못 재지만 **중력 방향**을 안다. 그래서:

1. IMU 의 `projected_gravity_b`(IMU 프레임에서 본 중력 단위벡터)를 외부 파라미터
   `R_cam<-imu` 로 카메라 광학 프레임으로 옮긴다 → 카메라 프레임의 중력 방향.
2. 뎁스에서 복원한 점마다 **중력 기준 높이** h = (설치 높이) + p·(-g) 를 구한다.
3. h 가 바닥 여유(floor_margin) 이하인 점은 바닥이므로 **버린다.**
4. 남은 점의 중력 정렬 전방 거리 d 중 최솟값으로 경고 등급을 정한다.
5. IMU 의 기울기·충격은 그 자체로 상위 정지 조건이 된다.

좌표계와 단위 (과제 노트에 적을 내용)
-----------------------------------
- 뎁스: `distance_to_image_plane`, 단위 **m**, 카메라 광학 z축 기준. 측정 못 한
  픽셀은 `inf` 로 온다(`CameraCfg.depth_clipping_behavior="none"` 기본값).
- 카메라 광학 프레임(ROS 규약): **+X 오른쪽, +Y 아래, +Z 시선**.
- IMU 프레임: `imu_link`. RosPider URDF 에서 base_link 대비 yaw -90° 회전.
- 중력: 단위벡터(크기 1), 가속도: m/s^2, 각도: 내부 계산은 rad, 출력은 deg.
- 두 센서는 같은 물리 스텝에서 읽는다. `update_period=0` 이면 매 스텝 갱신이므로
  시점 차이가 없다. 카메라만 `update_period` 를 키우면 뎁스가 몇 스텝 묵은 값이
  되므로, 기울기가 빠르게 변하는 구간에서는 융합 결과가 밀린다(노트의 '한계' 항목).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# 판정 코드. 숫자가 클수록 위험하다.
CLEAR, WARN, DANGER, STOP_TILT, STOP_IMPACT = 0, 1, 2, 3, 4
DECISION_NAMES = {
    CLEAR: "CLEAR",
    WARN: "WARN",
    DANGER: "DANGER",
    STOP_TILT: "STOP_TILT",
    STOP_IMPACT: "STOP_IMPACT",
}


@dataclass
class FusionParams:
    """융합에 들어가는 상수. 캘리브레이션 값과 임계값을 한 곳에 모았다."""

    # --- 외부 파라미터(캘리브레이션) ---
    r_cam_imu: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, -0.409498, -0.912311),
        (0.0, 0.912311, -0.409498),
    )
    """IMU 프레임 -> 카메라 광학 프레임 회전. `urdf_fk.py` 가 URDF 에서 뽑아준 값.

    기본값은 팔을 관측 자세(joint2=0.85, joint3=-1.60, joint4=-1.26) 로 고정했을 때.
    팔 자세나 카메라 장착을 바꾸면 **반드시 다시 뽑아야 한다.**
    """

    floor_mode: str = "estimated"
    """바닥 높이를 어떻게 잡을지.

    - ``"estimated"`` (기본): 보이는 점들의 하위 백분위수를 바닥으로 본다.
      설치 높이를 몰라도 되고, 몸체가 기울어 카메라 높이가 변해도 따라간다.
      대신 **바닥이 화면에 안 보이면 틀린다**(천장만 보는 경우 등).
    - ``"assumed_height"``: 설치 높이 :attr:`cam_height_m` 를 상수로 믿는다.
      단순하지만 기울어지면 그만큼 높이가 전부 밀린다. 둘을 비교하면 노션 노트의
      '한계/개선 방향' 항목에 쓸 재료가 나온다.
    """

    ground_percentile: float = 0.05
    """``floor_mode="estimated"`` 일 때 바닥으로 볼 하위 백분위수."""

    cam_height_m: float = 0.367
    """바닥에서 카메라 광학 중심까지의 설치 높이 [m].
    ``floor_mode="assumed_height"`` 에서만 쓴다.
    base_link 높이 0.11607 m + 카메라 z 오프셋 0.25086 m 가 수평 자세에서의 값이다."""

    # --- 관심 영역 ---
    corridor_half_width_m: float = 0.15
    """진행 경로의 반폭. 이보다 옆으로 벗어난 점은 지나갈 때 안 닿으므로 무시."""

    floor_margin_m: float = 0.03
    """이 높이 이하는 바닥으로 보고 버린다. RosPider 가 넘어갈 수 있는 단차이기도 하다."""

    ceiling_m: float = 0.40
    """이 높이 이상은 머리 위로 지나가므로 무시."""

    d_valid_min_m: float = 0.12
    """이 거리보다 가까운 점은 버린다 — **로봇 자기 몸을 지우는 값이다.**

    카메라가 팔 끝(link4)에 달려 있고 시선축이 link4 의 +Z 인데, link5 와 그리퍼가
    정확히 그 방향으로 뻗어 있다. 그래서 **그리퍼가 화면 중앙 아래에 늘 찍힌다.**
    URDF 로 재 보면 카메라 광학 프레임 기준:

    | 부위 | 전방 z | 횡 x | 그 거리에서 보이는 반폭 |
    |---|---|---|---|
    | link5 / gripper_link | -0.01 m | -0.003 m | (뒤쪽) |
    | l_link / r_link      |  0.027 m | ±0.015 m | 0.019 m -> 화면 안 |
    | 그리퍼 손가락         |  0.069 m | ±0.025 m | 0.048 m -> 화면 안 |
    | end_effector_link    |  0.070 m | -0.003 m | 0.049 m -> 화면 중앙 |

    이걸 안 지우면 융합이 늘 "앞 5 cm 에 장애물" 이라고 답한다(실제로 그렇게 나왔다).
    앞다리는 카메라 아래 0.22 m 라 시야 밖이므로 문제가 안 된다.
    0.12 m 로 두면 제일 먼 자기 부위(0.070 m)보다 넉넉히 뒤다.

    실기에서도 같은 처리를 한다(자기 몸 마스킹). 대신 **12 cm 안쪽은 원리적으로 못
    본다** — 노트의 '한계' 항목에 적을 것.
    """

    d_valid_max_m: float = 2.0
    """이 거리보다 먼 점은 버린다. 뒷벽과 노이즈를 걸러낸다."""

    min_obstacle_px: int = 25
    """장애물로 인정할 최소 픽셀 수. 한두 픽셀 튀는 걸로 경고가 뜨지 않게 한다."""

    # --- 판정 임계값 ---
    d_danger_m: float = 0.30
    d_warn_m: float = 0.60
    tilt_stop_deg: float = 25.0
    bump_acc_mps2: float = 6.0

    # --- 단일 센서 비교용 ---
    naive_center_frac: float = 0.4
    """뎁스 단독 기준선이 볼 중앙 열의 비율. 융합 쪽 corridor 에 대응."""

    roi_top_frac: float = 0.5
    """두 번째 기준선('뎁스 + 고정 ROI')이 쓸 상단 행의 비율.

    바닥을 빼려고 화면 위쪽만 잘라 쓰는, 현장에서 흔한 수동 보정이다.
    수평에서는 잘 돌지만 **몸체가 숙여지면 잘라낸 영역 안으로 바닥이 들어온다.**
    IMU 가 왜 필요한지 보여주는 비교군이라 융합 결과와 같이 기록한다."""

    device: str = field(default="cpu", repr=False)


@dataclass
class FusionResult:
    """한 스텝의 융합 결과. 전부 배치(N,) 또는 (N,H,W) 다."""

    decision: torch.Tensor
    """(N,) int64. CLEAR/WARN/DANGER/STOP_TILT/STOP_IMPACT 코드."""
    d_fused: torch.Tensor
    """(N,) 융합 거리 [m]. 장애물 없으면 inf."""
    d_naive: torch.Tensor
    """(N,) 뎁스 단독 거리 [m]. 바닥 제거도, 기울기 보정도 안 한 값."""
    decision_naive: torch.Tensor
    """(N,) 뎁스 단독으로만 판정했을 때의 코드. 융합과 비교하려고 같이 낸다."""
    d_roi: torch.Tensor
    """(N,) 뎁스 + 고정 ROI 기준선의 거리 [m]."""
    decision_roi: torch.Tensor
    """(N,) 뎁스 + 고정 ROI 기준선의 판정 코드."""
    tilt_deg: torch.Tensor
    """(N,) IMU 로 구한 몸체 기울기 [deg]."""
    bump_acc: torch.Tensor
    """(N,) IMU 선형가속 크기 [m/s^2]."""
    n_obstacle_px: torch.Tensor
    """(N,) 장애물로 분류된 픽셀 수."""
    obstacle_mask: torch.Tensor
    """(N,H,W) bool. 시각화용. 어느 픽셀이 최종 판단에 쓰였는지 보여준다."""
    height_map: torch.Tensor
    """(N,H,W) 바닥면 기준 높이 [m]. 바닥 픽셀은 0 근처로 나와야 한다."""

    def summary(self, env: int = 0) -> str:
        """로그 한 줄. 노션 노트의 실험 결과 표에 그대로 붙일 수 있다."""
        d_f = self.d_fused[env].item()
        d_n = self.d_naive[env].item()
        return (
            f"tilt={self.tilt_deg[env].item():5.1f}deg "
            f"acc={self.bump_acc[env].item():5.2f} "
            f"d_fused={d_f:6.3f} d_naive={d_n:6.3f} d_roi={self.d_roi[env].item():6.3f} "
            f"px={int(self.n_obstacle_px[env].item()):5d} | "
            f"fused={DECISION_NAMES[int(self.decision[env])]:11s} "
            f"naive={DECISION_NAMES[int(self.decision_naive[env])]:7s} "
            f"roi={DECISION_NAMES[int(self.decision_roi[env])]}"
        )


def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """뎁스 이미지를 카메라 광학 프레임의 점 집합으로 되돌린다.

    Args:
        depth: (N, H, W) 또는 (N, H, W, 1). `distance_to_image_plane`, 단위 m.
        intrinsics: (N, 3, 3) 카메라 내부 행렬. `camera.data.intrinsic_matrices`.

    Returns:
        (N, H, W, 3) 점 좌표 [m]. 광학 규약(+X 오른쪽, +Y 아래, +Z 시선).
        측정 실패 픽셀은 inf 가 그대로 전파된다.
    """
    if depth.dim() == 4:
        depth = depth[..., 0]
    n, h, w = depth.shape
    device = depth.device

    fx = intrinsics[:, 0, 0].view(n, 1, 1)
    fy = intrinsics[:, 1, 1].view(n, 1, 1)
    cx = intrinsics[:, 0, 2].view(n, 1, 1)
    cy = intrinsics[:, 1, 2].view(n, 1, 1)

    # 픽셀 중심 좌표. u 는 오른쪽, v 는 아래로 증가한다.
    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing="ij",
    )
    u = u.unsqueeze(0) + 0.5
    v = v.unsqueeze(0) + 0.5

    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack((x, y, z), dim=-1)


def _normalize(vec: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def gravity_in_camera(
    projected_gravity_b: torch.Tensor,
    params: FusionParams,
    r_cam_imu=None,
) -> torch.Tensor:
    """IMU 가 본 중력을 카메라 광학 프레임으로 옮긴다.

    Args:
        projected_gravity_b: (N, 3) `imu.data.projected_gravity_b`. 아래를 향하는 단위벡터.
        params: 임계값·상수. `r_cam_imu` 를 안 넘기면 여기 값을 쓴다.
        r_cam_imu: (3,3) 외부 파라미터. 팔이 움직이는 실험에서는 매 스텝 달라지므로
            `extrinsics.ArmCameraExtrinsics` 가 계산한 값을 여기로 넘긴다.

    Returns:
        (N, 3) 카메라 광학 프레임에서의 중력 단위벡터.
    """
    source = params.r_cam_imu if r_cam_imu is None else r_cam_imu
    rot = torch.as_tensor(
        source, dtype=projected_gravity_b.dtype, device=projected_gravity_b.device
    )
    return _normalize(projected_gravity_b @ rot.transpose(0, 1))


def fuse(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    lin_acc_b: torch.Tensor,
    params: FusionParams,
    r_cam_imu=None,
) -> FusionResult:
    """뎁스와 IMU 를 묶어 하나의 판정을 낸다.

    Args:
        depth: (N, H, W[, 1]) 뎁스 [m].
        intrinsics: (N, 3, 3) 내부 행렬.
        projected_gravity_b: (N, 3) IMU 프레임 중력 단위벡터.
        lin_acc_b: (N, 3) IMU 프레임 선형가속 [m/s^2].
            `ImuCfg.gravity_bias=(0,0,0)` 으로 두면 중력이 빠진 순수 운동가속이 된다.
        params: 임계값과 캘리브레이션 상수.
        r_cam_imu: (3,3) 외부 파라미터 override. 팔이 움직이면 매 스텝 바뀌므로
            `ArmCameraExtrinsics.r_cam_imu(joint_pos)` 결과를 넘긴다.
            None 이면 `params.r_cam_imu` 를 쓴다(팔 고정 자세용).
    """
    if depth.dim() == 4:
        depth = depth[..., 0]
    n, h, w = depth.shape

    # ---- 1. 뎁스 -> 점 집합 ----
    points = unproject_depth(depth, intrinsics)
    finite = torch.isfinite(depth) & (depth > 0)

    # ---- 2. IMU -> 카메라 프레임의 중력/상방/전방/우방 축 ----
    g_cam = gravity_in_camera(projected_gravity_b, params, r_cam_imu)  # (N,3) 아래 방향
    up = -g_cam
    ez = torch.zeros_like(up)
    ez[:, 2] = 1.0
    # 시선축에서 상방 성분을 뺀 것이 '중력에 정렬된 전방'이다.
    forward = _normalize(ez - (ez * up).sum(-1, keepdim=True) * up)
    right = _normalize(torch.cross(forward, up, dim=-1))

    # ---- 3. 중력 기준 높이 / 전방거리 / 횡거리 ----
    def project(axis: torch.Tensor) -> torch.Tensor:
        return (points * axis.view(n, 1, 1, 3)).sum(-1)

    up_coord = project(up)  # (N,H,W) 상방 좌표. 카메라가 원점이므로 바닥은 음수다.
    dist_fwd = project(forward)
    dist_lat = project(right)

    if params.floor_mode == "estimated":
        # 보이는 점의 하위 백분위수를 바닥면으로 본다. 설치 높이에 의존하지 않는다.
        valid = torch.where(finite, up_coord, torch.full_like(up_coord, float("nan")))
        ground = torch.nanquantile(valid.reshape(n, -1), params.ground_percentile, dim=1)
        ground = torch.nan_to_num(ground, nan=-params.cam_height_m)
        height = up_coord - ground.view(n, 1, 1)
    elif params.floor_mode == "assumed_height":
        height = params.cam_height_m + up_coord
    else:
        raise ValueError(f"floor_mode 는 'estimated' 또는 'assumed_height' 입니다: {params.floor_mode}")

    # ---- 4. 바닥/천장/경로밖 제거 ----
    mask = (
        finite
        & (height > params.floor_margin_m)
        & (height < params.ceiling_m)
        & (dist_lat.abs() < params.corridor_half_width_m)
        & (dist_fwd > params.d_valid_min_m)
        & (dist_fwd < params.d_valid_max_m)
    )
    n_px = mask.sum(dim=(1, 2))
    enough = n_px >= params.min_obstacle_px

    inf = torch.full((n,), float("inf"), dtype=depth.dtype, device=depth.device)
    d_masked = torch.where(mask, dist_fwd, inf.view(n, 1, 1))
    d_fused = torch.where(enough, d_masked.amin(dim=(1, 2)), inf)

    # ---- 5. IMU 단독 지표 ----
    # 수평일 때 projected_gravity_b = (0,0,-1). 그 방향과의 각도가 몸체 기울기다.
    g_down_z = (-_normalize(projected_gravity_b)[:, 2]).clamp(-1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(g_down_z))
    bump_acc = lin_acc_b.norm(dim=-1)

    # ---- 6. 최종 판정: 두 센서가 같이 쓰인다 ----
    decision = torch.full((n,), CLEAR, dtype=torch.long, device=depth.device)
    decision = torch.where(d_fused < params.d_warn_m, torch.full_like(decision, WARN), decision)
    decision = torch.where(d_fused < params.d_danger_m, torch.full_like(decision, DANGER), decision)
    # IMU 쪽 상위 조건이 거리 판정을 덮는다.
    decision = torch.where(bump_acc > params.bump_acc_mps2, torch.full_like(decision, STOP_IMPACT), decision)
    decision = torch.where(tilt_deg > params.tilt_stop_deg, torch.full_like(decision, STOP_TILT), decision)

    # ---- 7. 비교용 단일 센서 기준선 두 개 ----
    half = max(int(w * params.naive_center_frac / 2), 1)
    c0, c1 = w // 2 - half, w // 2 + half

    def grade(dist: torch.Tensor) -> torch.Tensor:
        out = torch.full((n,), CLEAR, dtype=torch.long, device=depth.device)
        out = torch.where(dist < params.d_warn_m, torch.full_like(out, WARN), out)
        return torch.where(dist < params.d_danger_m, torch.full_like(out, DANGER), out)

    def min_depth(rows: slice) -> torch.Tensor:
        patch = depth[:, rows, c0:c1]
        ok = finite[:, rows, c0:c1]
        return torch.where(ok, patch, inf.view(n, 1, 1)).amin(dim=(1, 2))

    # (a) 뎁스 단독: 중앙 밴드 최소 뎁스. 바닥도 장애물로 센다.
    d_naive = min_depth(slice(None))
    decision_naive = grade(d_naive)
    # (b) 뎁스 + 고정 ROI: 화면 위쪽만 봐서 바닥을 피하는 수동 보정.
    roi_rows = max(int(h * params.roi_top_frac), 1)
    d_roi = min_depth(slice(0, roi_rows))
    decision_roi = grade(d_roi)

    return FusionResult(
        decision=decision,
        d_fused=d_fused,
        d_naive=d_naive,
        decision_naive=decision_naive,
        d_roi=d_roi,
        decision_roi=decision_roi,
        tilt_deg=tilt_deg,
        bump_acc=bump_acc,
        n_obstacle_px=n_px,
        obstacle_mask=mask,
        height_map=height,
    )
