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
