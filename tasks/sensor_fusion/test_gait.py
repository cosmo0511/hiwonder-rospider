#!/usr/bin/env python3
"""gait.py 단위 테스트. Isaac Sim 없이 돌아간다 (numpy 만 필요).

제일 중요한 검사는 **평면 모델이 URDF 와 같은 답을 내는가** 다. `gait.py` 는 다리를
2링크 평면 문제로 줄여 풀지만, 진짜 기구학은 `urdf_fk.link_pose` 가 URDF 체인을
그대로 따라가며 계산한다. 둘이 어긋나면 보행이 미묘하게 틀어지는데, 시뮬 안에서는
"로봇이 좀 이상하게 걷네" 로만 보여서 원인을 찾기 어렵다. 그래서 여기서 못 박는다.

실행:
    python tasks/sensor_fusion/test_gait.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gait import (  # noqa: E402
    LEGS,
    TRIPOD_A,
    TRIPOD_B,
    GaitParams,
    HexapodGait,
    advance_phase,
    joint_name,
)
from urdf_fk import link_pose, load_joints  # noqa: E402

URDF = Path("assets/rospider_description/rospider.urdf")

# URDF 의 다리 조인트 한계(여섯 다리가 같다). 보행이 이 안에 들어야 한다.
JOINT_LIMITS = {
    "coxa": (-1.0, 1.5),
    "femur": (-1.75, 1.75),
    "tibla": (-1.75, 1.35),
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else '실패'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def urdf_foot(joints: dict, leg: str, t1: float, t2: float, t3: float) -> np.ndarray:
    """URDF 체인을 그대로 따라간 발 위치 (base_link 기준). 이게 정답이다."""
    q = {
        joint_name("coxa", leg): t1,
        joint_name("femur", leg): t2,
        joint_name("tibla", leg): t3,
    }
    pos, _ = link_pose(f"end_{leg}", joints, q, base="base_link")
    return pos


def main() -> int:
    if not URDF.exists():
        print(f"{URDF} 가 없습니다. 먼저:\n"
              "  bash tasks/sensor_fusion/00_fetch_description.sh\n"
              "  python tasks/sensor_fusion/01_xacro_to_urdf.py")
        return 1

    joints = load_joints(URDF)
    gait = HexapodGait(URDF)
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- A. 모델 일치
    print("\nA. 평면 모델 vs URDF 체인 — 같은 조인트각에서 같은 발 위치가 나오는가")
    worst = 0.0
    worst_at = ""
    for leg in LEGS:
        for _ in range(60):
            t1 = float(rng.uniform(-0.5, 0.5))
            t2 = float(rng.uniform(-0.6, 0.6))
            t3 = float(rng.uniform(-0.6, 0.6))
            err = float(np.linalg.norm(gait.legs[leg].fk(t1, t2, t3) - urdf_foot(joints, leg, t1, t2, t3)))
            if err > worst:
                worst, worst_at = err, f"{leg} ({t1:+.2f}, {t2:+.2f}, {t3:+.2f})"
    # 평면 모델은 횡방향 1.5 mm 오프셋을 무시한다. 그 이상 벌어지면 모델이 틀린 것이다.
    check("평면 모델이 URDF 와 3 mm 안에서 일치", worst < 0.003,
          f"최대 오차 {worst * 1000:.2f} mm @ {worst_at}")

    # ---------------------------------------------------------------- B. IK 왕복
    print("\nB. IK 왕복 — 목표점을 넣어 나온 각도를 URDF 에 다시 넣으면 그 점으로 돌아오는가")
    # 샘플 범위는 **보행이 실제로 쓰는 범위**다. 수평은 최대 보폭의 절반, 수직은
    # 발 드는 높이. 이보다 멀리 잡으면 작업공간 밖이라 IK 가 포화되는 게 정상이고,
    # 그걸 오차로 세면 안 된다(그 동작은 아래에서 따로 확인한다).
    reach_h = gait.params.max_stride_m / 2.0
    worst = 0.0
    worst_at = ""
    for leg in LEGS:
        n = gait.neutral_foot(leg)
        for _ in range(80):
            target = n + np.array([
                rng.uniform(-reach_h, reach_h),
                rng.uniform(-reach_h, reach_h),
                rng.uniform(-0.005, gait.params.step_height_m),
            ])
            t1, t2, t3 = gait.legs[leg].ik(target)
            err = float(np.linalg.norm(urdf_foot(joints, leg, t1, t2, t3) - target))
            if err > worst:
                worst, worst_at = err, f"{leg} {np.round(target, 4)}"
    check("IK -> URDF FK 왕복 오차 3 mm 이내", worst < 0.003,
          f"최대 오차 {worst * 1000:.2f} mm @ {worst_at}")

    print("\n  보행이 쓰는 범위가 전부 다리 작업공간 안에 들어오는가")
    tightest = 9.9
    tightest_at = ""
    for leg in LEGS:
        k = gait.legs[leg]
        for cmd_try in ((0.12, 0, 0), (-0.12, 0, 0), (0, 0.12, 0), (0, 0, 0.8), (0.08, 0.08, 0.4)):
            for i in range(24):
                p = gait.foot_target(leg, i / 24.0, cmd_try) - k.center
                rho = math.hypot(p[0], p[1])
                r = math.sqrt(max(rho * rho - k.y_off * k.y_off, 0.0))
                d = math.hypot(r - k.l1, p[2] - k.z0)
                margin = (k.l2 + k.l3) - d
                if margin < tightest:
                    tightest, tightest_at = margin, f"{leg} cmd={cmd_try}"
    # 여유를 5 mm 요구한다. 0 보다만 크면 된다고 두면 파라미터를 조금만 키워도
    # 포화 직전에서 아슬아슬하게 통과해 버린다(stance_radius 0.135 에서 3 mm).
    check("보행 궤적이 작업공간 안, 여유 5 mm 이상", tightest > 0.005,
          f"가장 뻗은 지점 여유 {tightest * 1000:.1f} mm — {tightest_at}")

    print("\n  닿을 수 없는 점을 넣어도 터지지 않고 포화되는가")
    far = gait.neutral_foot("LF") + np.array([1.0, 0.0, 0.0])
    try:
        angles = gait.legs["LF"].ik(far)
        ok = all(math.isfinite(a) for a in angles)
    except Exception as exc:  # noqa: BLE001
        ok = False
        angles = exc
    check("작업공간 밖 목표점이 유한한 각도로 포화", ok, f"{angles}")

    # ---------------------------------------------------------------- C. 보행 불변식
    print("\nC. 삼각보행 불변식")
    cmd = (0.10, 0.0, 0.0)
    phases = [i / 48.0 for i in range(48)]

    # C1. 항상 세 다리가 땅에 있다.
    always_three = all(len(gait.swinging(p)) == 3 for p in phases)
    check("언제나 정확히 세 다리가 공중에 있다", always_three)

    # C2. 두 조가 절대 같이 뜨지 않는다.
    overlap = any(set(gait.swinging(p)) not in (set(TRIPOD_A), set(TRIPOD_B)) for p in phases)
    check("뜨는 조합은 항상 A조 또는 B조", not overlap)

    # C3. 딛는 발은 지면 높이에 붙어 있고, 나는 발만 들린다.
    stance_z, swing_z = [], []
    for p in phases:
        flying = gait.swinging(p)
        for leg in LEGS:
            z = gait.foot_target(leg, p, cmd)[2]
            (swing_z if leg in flying else stance_z).append(z)
    ground = -gait.params.body_height_m
    check("딛는 발은 지면 높이에 고정", max(abs(z - ground) for z in stance_z) < 1e-9,
          f"최대 이탈 {max(abs(z - ground) for z in stance_z) * 1000:.4f} mm")
    check("나는 발이 설정한 만큼 들린다",
          abs(max(swing_z) - (ground + gait.params.step_height_m)) < 1e-6,
          f"최대 {max(swing_z) - ground:.4f} m (설정 {gait.params.step_height_m} m)")

    # C4. 명령이 0 이면 모든 발이 중립점에 가만히 있다(들리지도 않는다 — 제자리걸음 금지).
    still = max(
        float(np.linalg.norm(gait.foot_target(leg, p, (0.0, 0.0, 0.0)) - gait.neutral_foot(leg)))
        for leg in LEGS
        for p in phases
    )
    check("명령 0 에서는 발이 중립점을 벗어나지 않는다", still < 1e-9,
          f"최대 이동 {still * 1000:.4f} mm")

    # ---------------------------------------------------------------- D. 방향 부호
    print("\nD. 명령 방향이 실제로 그 방향인가")
    # 전진 명령이면, 딛는 동안 발은 몸 기준으로 **뒤로** 쓸려야 한다.
    swept = []
    for leg in LEGS:
        local_start = 0.0 if leg in TRIPOD_A else 0.5
        p0 = local_start
        p1 = (local_start + gait.params.duty * 0.9) % 1.0
        swept.append(gait.foot_target(leg, p1, (0.10, 0.0, 0.0))[0]
                     - gait.foot_target(leg, p0, (0.10, 0.0, 0.0))[0])
    check("전진 명령에서 딛는 발이 전부 뒤로 쓸린다", all(s < -0.005 for s in swept),
          f"x 변화 {np.round(swept, 4)}")

    # 좌측 이동 명령이면 발은 우측(−Y)으로 쓸린다.
    swept_y = []
    for leg in LEGS:
        local_start = 0.0 if leg in TRIPOD_A else 0.5
        p1 = (local_start + gait.params.duty * 0.9) % 1.0
        swept_y.append(gait.foot_target(leg, p1, (0.0, 0.08, 0.0))[1]
                       - gait.foot_target(leg, local_start, (0.0, 0.08, 0.0))[1])
    check("좌측 명령에서 딛는 발이 전부 우측으로 쓸린다", all(s < -0.004 for s in swept_y),
          f"y 변화 {np.round(swept_y, 4)}")

    # 제자리 좌회전이면 좌/우 다리가 서로 반대 방향으로 쓸린다.
    turn = {}
    for leg in ("LM", "RM"):
        local_start = 0.0 if leg in TRIPOD_A else 0.5
        p1 = (local_start + gait.params.duty * 0.9) % 1.0
        turn[leg] = (gait.foot_target(leg, p1, (0.0, 0.0, 0.6))
                     - gait.foot_target(leg, local_start, (0.0, 0.0, 0.6)))[0]
    check("제자리 회전에서 좌우 다리가 반대로 쓸린다", turn["LM"] * turn["RM"] < 0,
          f"LM dx={turn['LM']:+.4f}  RM dx={turn['RM']:+.4f}")

    # ---------------------------------------------------------------- E. 보폭 상한
    print("\nE. 보폭 상한과 속도 상한")
    fast = gait.half_stride("LF", (10.0, 0.0, 0.0))
    check("과한 속도 명령에도 보폭이 상한에 묶인다",
          float(np.linalg.norm(fast)) <= gait.params.max_stride_m / 2.0 + 1e-9,
          f"절반 보폭 {np.linalg.norm(fast) * 1000:.1f} mm (상한 {gait.params.max_stride_m * 500:.1f} mm)")
    clipped = gait.clip_cmd((10.0, 10.0, 10.0))
    check("속도 명령이 상한으로 눌린다",
          math.hypot(clipped[0], clipped[1]) <= gait.params.max_v_mps + 1e-9
          and abs(clipped[2]) <= gait.params.max_w_radps + 1e-9,
          f"{np.round(clipped, 3)}")

    # ---------------------------------------------------------------- F. 조인트 한계
    print("\nF. 한 주기 내내 조인트 한계 안에 머무는가 (URDF limit)")
    worst_margin = 9.9
    worst_label = ""
    for cmd_try in ((0.12, 0, 0), (-0.12, 0, 0), (0, 0.12, 0), (0, 0, 0.8), (0.08, 0.08, 0.4)):
        for p in phases:
            for name, value in gait.joint_targets(p, cmd_try).items():
                lo, hi = JOINT_LIMITS[name.split("_")[0]]
                margin = min(value - lo, hi - value)
                if margin < worst_margin:
                    worst_margin, worst_label = margin, f"{name}={value:+.3f} (한계 {lo}~{hi})"
    check("모든 조인트가 한계 안", worst_margin > 0.0,
          f"가장 빠듯한 곳 여유 {worst_margin:.3f} rad — {worst_label}")

    # ---------------------------------------------------------------- G. 위상 진행
    print("\nG. 위상 진행")
    p = 0.0
    for _ in range(int(GaitParams().period_s / 0.01)):
        p = advance_phase(p, 0.01, GaitParams().period_s)
    check("한 주기를 돌면 위상이 제자리로", abs(p) < 1e-6 or abs(p - 1.0) < 1e-6, f"phase={p:.6f}")
    check("위상은 항상 [0,1)", 0.0 <= advance_phase(0.99, 0.5, 1.0) < 1.0)

    # ---------------------------------------------------------------- H. 몸통 흔들림
    print("\nH. 운동학 모드의 몸통 흔들림 모델")
    still_sway = max(abs(v) for v in gait.body_sway(0.3, (0.0, 0.0, 0.0)))
    check("서 있으면 흔들리지 않는다", still_sway < 1e-12, f"최대 {still_sway:.2e}")
    pitches = [gait.body_sway(p, (0.12, 0.0, 0.0))[1] for p in phases]
    check("걸으면 몸이 설정한 진폭만큼 끄덕인다",
          abs(max(pitches) - math.radians(gait.params.body_pitch_deg)) < 1e-3,
          f"pitch 진폭 {math.degrees(max(pitches)):.2f} deg "
          f"(설정 {gait.params.body_pitch_deg} deg)")
    check("한 주기 평균 pitch 는 0 (한쪽으로 쏠리지 않는다)",
          abs(float(np.mean(pitches))) < 1e-9, f"평균 {np.mean(pitches):.2e} rad")

    print("\n" + "=" * 90)
    if failures:
        print(f"실패 {len(failures)}건:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
