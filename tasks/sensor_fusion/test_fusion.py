#!/usr/bin/env python3
"""fusion.py 단위 테스트. Isaac Sim 없이 돌아간다.

Kit 을 띄우지 않고 합성 뎁스를 직접 만들어 넣는다. 바닥 평면과 직육면체 하나를
해석적으로 레이캐스팅해 `distance_to_image_plane` 과 같은 성질의 뎁스를 만든다.
덕분에 **융합 로직의 버그와 시뮬레이터 문제를 분리**할 수 있다. Isaac 쪽을 건드리기
전에 항상 이걸 먼저 통과시키자.

실행:
    # 컨테이너 안, isaac-shell (torch 가 있으면 어디서든 됨)
    python tasks/sensor_fusion/test_fusion.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion import (  # noqa: E402
    CLEAR,
    DANGER,
    DECISION_NAMES,
    STOP_IMPACT,
    STOP_TILT,
    WARN,
    FusionParams,
    fuse,
)

# urdf_fk.py 가 뽑아준 값. base_link 기준 카메라 광학 프레임의 축과 위치.
# (팔 관측 자세 joint2=0.85, joint3=-1.60, joint4=-1.26)
CAM_OFFSET_IN_BASE = (0.13125, 0.0013, 0.25086)
R_BASE_CAM = (  # 열이 각각 카메라 X(right), Y(down), Z(view) 축
    (0.0, -0.409498, 0.912311),
    (-1.0, 0.0, 0.0),
    (0.0, -0.912311, -0.409498),
)
BASE_HEIGHT = 0.11607  # base_footprint -> base_link


def make_intrinsics(width: int, height: int, hfov_deg: float) -> torch.Tensor:
    """수평 화각으로 핀홀 내부 행렬을 만든다."""
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    k = torch.eye(3, dtype=torch.float64)
    k[0, 0] = fx
    k[1, 1] = fx
    k[0, 2] = width / 2
    k[1, 2] = height / 2
    return k.unsqueeze(0)


def rot_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def render_scene(
    width: int,
    height: int,
    intrinsics: torch.Tensor,
    pitch_deg: float,
    obstacle_dist: float | None,
    obstacle_height: float,
    obstacle_depth: float = 0.20,
    obstacle_half_width: float = 0.12,
    boxes: list[tuple[float, float, float, float, float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """합성 뎁스와 IMU 중력을 만든다.

    바닥 평면 z=0 과 직육면체 하나. `obstacle_dist` 는 **카메라 기준 수평거리**라서
    융합이 내놓는 d_fused 와 바로 비교할 수 있다.
    로봇은 base 원점 (0,0,BASE_HEIGHT) 에서 pitch 만큼 앞으로 숙인 자세로 고정.

    Returns:
        depth: (1, H, W) float64, 단위 m. 아무것도 안 맞으면 inf.
        projected_gravity_b: (1, 3) IMU 프레임 중력 단위벡터.
    """
    pitch = math.radians(pitch_deg)
    r_w_base = rot_y(pitch)
    r_base_cam = torch.tensor(R_BASE_CAM, dtype=torch.float64)
    r_w_cam = r_w_base @ r_base_cam
    cam_pos = torch.tensor([0.0, 0.0, BASE_HEIGHT], dtype=torch.float64) + r_w_base @ torch.tensor(
        CAM_OFFSET_IN_BASE, dtype=torch.float64
    )

    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    cx = intrinsics[0, 0, 2]
    cy = intrinsics[0, 1, 2]
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    # 뎁스 s 로 매개화한 광선. 광학 프레임에서 p = s * (x', y', 1).
    ray_cam = torch.stack(((u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones_like(u)), dim=-1)
    ray_w = ray_cam @ r_w_cam.transpose(0, 1)  # (H, W, 3)

    inf = torch.full((height, width), float("inf"), dtype=torch.float64)

    # --- 바닥 z=0 ---
    denom = ray_w[..., 2]
    s_ground = torch.where(denom.abs() > 1e-12, -cam_pos[2] / denom, inf)
    s_ground = torch.where(s_ground > 1e-6, s_ground, inf)

    # --- 상자들 (slab method) ---
    # boxes 는 카메라 기준 상대좌표 (dx0, dx1, y0, y1, z0, z1). 계단을 세울 때 쓴다.
    aabbs: list[tuple[float, float, float, float, float, float]] = []
    if boxes is not None:
        aabbs.extend(boxes)
    elif obstacle_dist is not None:
        aabbs.append(
            (obstacle_dist, obstacle_dist + obstacle_depth,
             -obstacle_half_width, obstacle_half_width, 0.0, obstacle_height)
        )

    s_box = inf
    for dx0, dx1, y0, y1, z0, z1 in aabbs:
        lo = torch.tensor([cam_pos[0].item() + dx0, y0, z0], dtype=torch.float64)
        hi = torch.tensor([cam_pos[0].item() + dx1, y1, z1], dtype=torch.float64)
        t_lo = (lo - cam_pos) / ray_w
        t_hi = (hi - cam_pos) / ray_w
        t_near = torch.minimum(t_lo, t_hi).amax(dim=-1)
        t_far = torch.maximum(t_lo, t_hi).amin(dim=-1)
        hit = (t_far >= t_near.clamp_min(0.0)) & (t_far > 0)
        s_box = torch.minimum(s_box, torch.where(hit, t_near.clamp_min(0.0), inf))

    depth = torch.minimum(s_ground, s_box).unsqueeze(0)

    # --- IMU: imu_link 는 base_link 대비 yaw -90도 ---
    r_w_imu = r_w_base @ rot_z(-math.pi / 2)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    projected_gravity_b = (r_w_imu.transpose(0, 1) @ gravity_w).unsqueeze(0)
    return depth, projected_gravity_b


def run_case(
    name: str,
    pitch_deg: float,
    obstacle_dist: float | None,
    obstacle_height: float,
    acc: tuple[float, float, float] = (0.0, 0.0, 0.0),
    params: FusionParams | None = None,
    width: int = 160,
    height: int = 120,
    boxes: list | None = None,
) -> tuple:
    intrinsics = make_intrinsics(width, height, hfov_deg=70.0)
    depth, g = render_scene(
        width, height, intrinsics, pitch_deg, obstacle_dist, obstacle_height, boxes=boxes
    )
    lin_acc = torch.tensor([acc], dtype=torch.float64)
    result = fuse(depth, intrinsics, g, lin_acc, params or FusionParams())
    print(f"{name:34s} {result.summary()}")
    return result


def stairs_boxes(
    x_front: float = 0.55,
    n_steps: int = 4,
    step_height: float = 0.05,
    step_depth: float = 0.14,
    half_width: float = 0.35,
) -> list[tuple[float, float, float, float, float, float]]:
    """rospider_cfg._build_stairs 와 같은 치수의 계단을 합성 씬에 세운다."""
    return [
        (
            x_front + step_depth * i,
            x_front + step_depth * (i + 1),
            -half_width,
            half_width,
            0.0,
            step_height * (i + 1),
        )
        for i in range(n_steps)
    ]


def main() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)
            print(f"    [실패] {label}")

    print("=" * 118)
    print("조건 A: 수평, 장애물 0.80 m(카메라 기준), 높이 0.15 m")
    r = run_case("A 수평 + 장애물 0.80 m", 0.0, 0.80, 0.15)
    check("A: 융합 거리 0.80 m ±0.05", abs(r.d_fused.item() - 0.80) < 0.05)
    check("A: 융합 CLEAR", int(r.decision) == CLEAR)
    check("A: 고정 ROI 기준선도 CLEAR (수평에서는 통함)", int(r.decision_roi) == CLEAR)
    check("A: 뎁스 단독은 바닥 때문에 경고", int(r.decision_naive) in (WARN, DANGER))
    check("A: IMU 기울기 ~0도", r.tilt_deg.item() < 0.5)

    print("\n조건 B: pitch 20도 숙임, 장애물 위치 동일 -- 고정 ROI 가 깨지는 지점")
    r = run_case("B 숙임 20도 + 장애물 0.80 m", 20.0, 0.80, 0.15)
    check("B: 융합 거리 0.80 m 유지 ±0.06", abs(r.d_fused.item() - 0.80) < 0.06)
    check("B: 융합 CLEAR 유지", int(r.decision) == CLEAR)
    check("B: 고정 ROI 는 바닥을 장애물로 오인", int(r.decision_roi) in (WARN, DANGER))
    check("B: IMU 기울기 20도 측정", abs(r.tilt_deg.item() - 20.0) < 0.5)

    print("\n조건 C: 수평, 장애물이 0.55 m -> 0.25 m 로 접근 -- 경고 등급 전이")
    r1 = run_case("C1 수평 + 장애물 0.55 m", 0.0, 0.55, 0.15)
    r2 = run_case("C2 수평 + 장애물 0.25 m", 0.0, 0.25, 0.15)
    check("C1: WARN", int(r1.decision) == WARN)
    check("C2: DANGER", int(r2.decision) == DANGER)
    check("C1->C2 거리 감소", r2.d_fused.item() < r1.d_fused.item())

    print("\n조건 D: 수평, 2 cm 단차만 있음 -- 넘어갈 수 있으니 장애물이 아니다")
    r = run_case("D 수평 + 단차 0.02 m @0.55 m", 0.0, 0.55, 0.02)
    check("D: 융합은 장애물 없음", math.isinf(r.d_fused.item()))
    check("D: 융합 CLEAR", int(r.decision) == CLEAR)

    print("\n조건 E: 30도까지 넘어짐 -- IMU 가 거리 판정을 덮는다")
    r = run_case("E 숙임 30도", 30.0, 0.80, 0.15)
    check("E: STOP_TILT", int(r.decision) == STOP_TILT)

    print("\n조건 F: 수평 + 충격 가속 8 m/s^2 -- 충격 정지가 최우선")
    r = run_case("F 수평 + 충격 8 m/s^2", 0.0, 0.80, 0.15, acc=(0.0, 0.0, 8.0))
    check("F: STOP_IMPACT", int(r.decision) == STOP_IMPACT)

    print("\n조건 G: 같은 B 를 floor_mode='assumed_height' 로 -- 설치높이 상수의 한계")
    r = run_case("G 숙임 20도 + 설치높이 상수", 20.0, 0.80, 0.15, params=FusionParams(floor_mode="assumed_height"))
    print(
        f"    (참고) 숙이면 카메라가 낮아지는데 상수를 믿으므로 높이가 밀린다 -> "
        f"d_fused={r.d_fused.item():.3f} m, 판정 {DECISION_NAMES[int(r.decision)]}"
    )

    print("\n조건 H: 계단 4단 (단 높이 5 cm). 높이맵이 띠로 갈라져야 한다")
    stairs = stairs_boxes()
    r = run_case("H 수평 + 계단", 0.0, None, 0.0, boxes=stairs)
    hm = r.height_map[0][r.obstacle_mask[0]]
    check("H: 첫 단을 0.55 m 근처에서 잡음", abs(r.d_fused.item() - 0.55) < 0.06)
    check("H: WARN", int(r.decision) == WARN)
    if hm.numel():
        bands = torch.histc(hm, bins=5, min=0.0, max=0.25)
        print(f"    높이 분포(0~0.25 m 를 5칸): {[int(v) for v in bands]}  "
              f"범위 {hm.min():.3f}~{hm.max():.3f} m")
        check("H: 여러 단 높이가 섞여 나옴(띠)", int((bands > 20).sum()) >= 3)

    print("\n조건 I: 같은 계단, 몸체 15도 숙임. 보이는 단의 높이 추정이 유지돼야 한다")
    r2 = run_case("I 숙임 15도 + 계단", 15.0, None, 0.0, boxes=stairs)
    hm2 = r2.height_map[0][r2.obstacle_mask[0]]
    bins = [(0.00, 0.06, "1단 0.05"), (0.06, 0.11, "2단 0.10"),
            (0.11, 0.16, "3단 0.15"), (0.16, 0.22, "4단 0.20")]
    level = [int(((hm >= lo) & (hm < hi)).sum()) for lo, hi, _ in bins]
    tilted = [int(((hm2 >= lo) & (hm2 < hi)).sum()) for lo, hi, _ in bins]
    for (_, _, name), a, b in zip(bins, level, tilted):
        print(f"    {name}: 수평 {a:5d} px -> 숙임 {b:5d} px")
    check("I: 거리 추정이 기울기와 무관하게 유지", abs(r2.d_fused.item() - r.d_fused.item()) < 0.02)
    check("I: 1~3단이 기울인 뒤에도 같은 높이 구간에 잡힘", all(v > 100 for v in tilted[:3]))
    # 4단이 사라지는 것은 융합 오류가 아니라 **시야(FOV) 한계**다. 하향 24도 장착에
    # 몸이 15도 더 숙으면 먼 곳/높은 곳이 화면 위로 잘려 나간다. 노트의 '한계' 항목.
    print(f"    (주의) 4단은 숙이면 시야 밖으로 나간다: {level[3]} px -> {tilted[3]} px. "
          f"융합 오류가 아니라 FOV 한계다.")

    print("\n조건 J: 자기 그리퍼가 화면에 찍힐 때 -- 기준선 전처리가 왜 필요한가")
    # 실기(Isaac) 실행에서 실제로 나온 문제다. 카메라가 팔 끝에 달려 있어 그리퍼가
    # 화면 중앙 아래에 늘 보이고, outputs/sensor_fusion/run_log.csv 를 보면
    # d_naive 가 장애물 거리와 무관하게 **0.0642 m 로 고정**돼 있다. 합성 씬에는
    # 로봇 몸이 없으니 여기서 직접 그려 넣어 같은 상황을 만든다.
    width, height = 160, 120
    intrinsics_j = make_intrinsics(width, height, hfov_deg=70.0)
    depth_j, g_j = render_scene(width, height, intrinsics_j, 0.0, 0.80, 0.15)
    gripper_depth = 0.064  # 실측 로그의 d_naive 값
    depth_j[0, int(height * 0.55):int(height * 0.80), int(width * 0.44):int(width * 0.56)] = (
        gripper_depth
    )
    lin_acc_j = torch.zeros((1, 3), dtype=torch.float64)

    r_on = fuse(depth_j, intrinsics_j, g_j, lin_acc_j, FusionParams())
    r_off = fuse(depth_j, intrinsics_j, g_j, lin_acc_j,
                 FusionParams(baseline_self_mask=False))
    print(f"{'J1 전처리 켬 (기본)':34s} {r_on.summary()}")
    print(f"{'J2 전처리 끔 (예전 동작)':34s} {r_off.summary()}")

    check("J: 융합은 자기 몸에 흔들리지 않는다", abs(r_on.d_fused.item() - 0.80) < 0.05)
    check("J: 융합 CLEAR 유지", int(r_on.decision) == CLEAR)
    check("J: 전처리를 켜면 기준선이 그리퍼를 무시한다",
          r_on.d_naive.item() > FusionParams().d_valid_min_m)
    check("J: 전처리를 켜면 기준선 값이 그리퍼 없는 조건 A 와 같다",
          abs(r_on.d_naive.item() - 0.415) < 0.02)
    check("J: 전처리를 끄면 기준선이 그리퍼를 장애물로 본다",
          abs(r_off.d_naive.item() - gripper_depth) < 0.005)
    check("J: 전처리를 끄면 기준선은 거리와 무관하게 DANGER",
          int(r_off.decision_naive) == DANGER)
    print("    -> 전처리를 끄면 d_naive 가 장애물이 아니라 자기 그리퍼를 가리킨다"
          f" ({r_off.d_naive.item():.4f} m). 그 상태로 비교하면 '융합이 낫다' 가 아니라"
          " '기준선이 고장' 이다.")

    print("\n조건 K: 걸으면서 장애물에 접근 -- 이 과제의 시나리오 그대로")
    # gait.py 의 보행 위상에서 몸통 끄덕임을 받아, 매 프레임 그 자세로 합성 뎁스를
    # 다시 렌더링한다. Isaac 없이 **보행 중 판정**을 통째로 돌려 보는 셈이다.
    # (Kit 단계로 넘어가기 전에 여기서 논리가 맞는지 먼저 확인한다 — 이 레포의 순서다.)
    try:
        from gait import GaitParams, HexapodGait, advance_phase  # noqa: PLC0415

        gait_urdf = Path("assets/rospider_description/rospider.urdf")
        if not gait_urdf.exists():
            raise FileNotFoundError(gait_urdf)
        gp = GaitParams()
        walker = HexapodGait(gait_urdf, gp)
    except Exception as exc:  # noqa: BLE001
        print(f"    [건너뜀] 보행 모듈을 못 불러왔다({exc}).")
        print("              bash tasks/sensor_fusion/00_fetch_description.sh 를 먼저 실행하세요.")
        walker = None

    if walker is not None:
        speed = 0.10          # m/s, 전진
        dt = 1.0 / 30.0       # 30 Hz 로 판정한다고 보자
        cmd_k = (speed, 0.0, 0.0)
        dist_k = 1.10         # 시작 거리 [m]
        phase_k = 0.0
        track_fused, track_roi, track_naive = [], [], []
        first_stop = {"fused": None, "roi": None, "naive": None}
        intrinsics_k = make_intrinsics(160, 120, hfov_deg=70.0)

        while dist_k > 0.18:
            phase_k = advance_phase(phase_k, dt, gp.period_s)
            _, sway_pitch, _ = walker.body_sway(phase_k, cmd_k)
            pitch_k = math.degrees(sway_pitch)
            depth_k, g_k = render_scene(160, 120, intrinsics_k, pitch_k, dist_k, 0.15)
            rk = fuse(depth_k, intrinsics_k, g_k,
                      torch.zeros((1, 3), dtype=torch.float64), FusionParams())
            track_fused.append(abs(rk.d_fused.item() - dist_k))
            track_roi.append(abs(rk.d_roi.item() - dist_k))
            track_naive.append(abs(rk.d_naive.item() - dist_k))
            for key, dec in (("fused", rk.decision), ("roi", rk.decision_roi),
                             ("naive", rk.decision_naive)):
                if first_stop[key] is None and int(dec) == DANGER:
                    first_stop[key] = dist_k
            dist_k -= speed * dt

        def worst(values):
            return max(v for v in values if math.isfinite(v))

        def fmt_stop(v):
            return "못 멈춤" if v is None else f"{v:.3f} m"

        print(f"    몸통 끄덕임 ±{gp.body_pitch_deg}도, {len(track_fused)} 프레임, "
              f"1.10 m -> 0.18 m 접근 (전진 {speed} m/s, 판정 {1 / dt:.0f} Hz)")
        print(f"    거리 오차(최대):  융합 {worst(track_fused) * 100:5.1f} cm   "
              f"고정ROI {worst(track_roi) * 100:5.1f} cm   "
              f"뎁스단독 {worst(track_naive) * 100:5.1f} cm")
        print(f"    DANGER 를 처음 낸 참거리 (임계값 0.300 m):  "
              f"융합 {fmt_stop(first_stop['fused'])}   "
              f"고정ROI {fmt_stop(first_stop['roi'])}   "
              f"뎁스단독 {fmt_stop(first_stop['naive'])}")
        if first_stop["roi"] is None:
            print("    -> 고정 ROI 는 끝까지 DANGER 를 못 냈다. **안 멈추는 쪽(false negative)** 이라"
                  " 오경보보다 위험하다.")
        if first_stop["naive"] is not None:
            late = 0.30 - first_stop["naive"]
            print(f"    -> 뎁스 단독은 {late * 100:.1f} cm 늦게 멈춘다. 최소 뎁스가 바닥까지의"
                  " **사선 거리**라 실제 전방거리보다 크게 읽히기 때문이다.")

        check("K: 보행 중에도 융합 거리가 참값을 5 cm 안에서 따라간다",
              worst(track_fused) < 0.05)
        check("K: 융합이 기준선보다 참값에 가깝다",
              worst(track_fused) < worst(track_roi)
              and worst(track_fused) < worst(track_naive))
        check("K: 융합은 임계값 0.30 m 근처에서 DANGER 를 낸다",
              first_stop["fused"] is not None
              and abs(first_stop["fused"] - 0.30) < 0.06)
        check("K: 기준선은 제때 못 멈추거나 엉뚱한 거리에서 멈춘다",
              first_stop["roi"] is None or abs(first_stop["roi"] - 0.30) > 0.06
              or first_stop["naive"] is None or abs(first_stop["naive"] - 0.30) > 0.06)

    print("\n" + "=" * 118)
    if failures:
        print(f"실패 {len(failures)}건:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
