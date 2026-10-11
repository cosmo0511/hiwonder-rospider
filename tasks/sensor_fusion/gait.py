"""RosPider 6족 삼각보행(tripod gait) 생성기. **numpy 만 쓴다 (Kit 불필요).**

`extrinsics.py` 와 같은 원칙이다 — 숫자를 손으로 적지 않고 **URDF 에서 뽑는다.**
`HexapodGait(urdf_path)` 를 만들면 생성자가 `urdf_fk` 로 여섯 다리의 체인을 풀어
링크 길이와 장착 각도를 계산한다. 로봇이 바뀌면 상수가 아니라 URDF 를 바꾼다.

왜 보행이 이 과제에 필요한가
----------------------------
이 레포의 융합(`fusion.py`)은 **IMU 로 뎁스 점을 중력에 정렬해 바닥을 지우는** 것이
핵심이다. 그런데 로봇이 가만히 서 있으면 기울기가 0 이라 IMU 가 할 일이 없다.
그래서 지금까지는 `--pitch_deg 20` 처럼 **사람이 손으로 몸을 숙여** 놓고 실험했다.
설득력이 약하다 — "그렇게 기울어질 일이 실제로 있나?" 라는 질문에 답을 못 한다.

6족 보행은 **걷는 것 자체가 몸을 끄덕이게 한다.** 세 다리씩 번갈아 딛으므로 지지
다각형이 매 반주기 바뀌고, 몸체 pitch/roll 이 주기적으로 흔들린다. 보행 중에
전방 판단을 하면 기울기는 **인위적 조건이 아니라 과제의 전제**가 된다.

기구학 (URDF 에서 확인한 구조)
------------------------------
여섯 다리가 **같은 구조**다(수치가 1.5 mm 이내로 일치). 다리마다:

    base_link ──(고정)── leg_center_XX ──coxa(θ1, 지역 +Z)── femur(θ2) ── tibla(θ3) ── end

- `leg_center_XX` 는 base_link 기준 고정 위치이고, coxa 조인트의 rpy yaw 가
  **그 다리가 몸통에서 뻗어 나가는 방향**(LF 45도, LM 90도, LR 135도, 우측은 부호 반대)이다.
- coxa θ1 은 지역 +Z 회전 = 수평면 안에서 다리를 좌우로 휘두른다.
- femur θ2 와 tibla θ3 은 **다리의 반경-수직 평면 안**에서만 움직인다. 발끝은 그
  평면에서 횡으로 `y_off` 만큼 비켜 있는데, 이 값은 **조인트각과 무관한 상수**다
  (세 각을 다 흔들어도 변하지 않는 것을 확인했다). 그래서 모델에 그대로 넣는다.
- θ2 가 **양수면 발이 올라간다. 여섯 다리 전부 같은 부호다** — URDF 작성자가 좌우
  대칭을 축이 아니라 rpy 로 처리해 둔 덕분이다. 그래서 부호 분기가 필요 없다.

> **RM 다리는 URDF 상 다른 다리와 다르다.** `end_RM_joint` 의 origin 이 나머지
> 다섯과 어긋나 있어 `y_off` 가 −17.4 mm 다(다른 다리는 ±1.5~3 mm). 발 위치도
> 4 mm 높다. Hiwonder URDF 쪽 불일치로 보이는데, 상수를 다리마다 따로 뽑으므로
> 이 레포에서는 그대로 맞춰 들어간다. **`y_off` 를 무시하면 RM 만 17 mm 틀린다**
> — `test_gait.py` 의 A/B 검사가 정확히 그걸 잡아낸다.

평면 2링크 모델이 **정확히** 맞는 것을 확인했다(`test_gait.py` 가 검사한다):

    r = L1 + L2·cos(θ2 + a2) + L3·cos(θ2 + θ3 + a3)
    z = z0 + L2·sin(θ2 + a2) + L3·sin(θ2 + θ3 + a3)

`a2`, `a3` 는 링크가 조인트 축과 정렬돼 있지 않아 생기는 오프셋 각이다. 이것 때문에
조인트를 전부 0 으로 두어도 무릎이 이미 굽어 있다(발이 바닥 아래 1.07 cm 에 온다).

삼각보행
--------
A조 = (LF, RM, LR), B조 = (RF, LM, RR). 한 조가 딛는 동안 다른 조가 난다.
B조의 위상은 A조 + 0.5 다. 명령 (vx, vy, wz) 에 대해 다리 i 의 보폭 벡터는

    s_i = (vx − wz·n_iy,  vy + wz·n_ix)        n_i = 그 다리의 중립 접지점

이고, 딛는 동안 발은 몸 기준으로 −s_i 방향으로 쓸려 간다(그만큼 몸이 전진한다).

단위
----
거리 m, 각도 rad(파라미터만 deg), 속도 m/s, 각속도 rad/s. 좌표는 전부 **base_link**
기준이며 +X 전방, +Y 좌측, +Z 상방이다(URDF 규약).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from urdf_fk import link_pose, load_joints

LEGS = ("LF", "LM", "LR", "RF", "RM", "RR")
"""L/R = 좌/우, F/M/R = 앞/중간/뒤. URDF 조인트 이름에 그대로 쓰인다."""

TRIPOD_A = ("LF", "RM", "LR")
TRIPOD_B = ("RF", "LM", "RR")
"""삼각보행의 두 조. 각 조의 세 발이 삼각형을 이뤄 몸을 받친다."""

JOINT_PREFIX = ("coxa", "femur", "tibla")
"""URDF 조인트 이름 규칙: ``{prefix}_{leg}_joint``. 'tibla' 는 URDF 의 철자 그대로다."""


def joint_name(prefix: str, leg: str) -> str:
    return f"{prefix}_{leg}_joint"


@dataclass
class GaitParams:
    """보행 손잡이. 전부 [미검증] 출발점이다 — 실기에서 맞춘 값이 아니다."""

    period_s: float = 1.2
    """한 보행 주기 [s]. 한 주기에 각 조가 한 번씩 딛고 한 번씩 난다."""

    duty: float = 0.5
    """한 주기 중 딛는 비율. 삼각보행은 0.5 다(두 조가 정확히 교대).
    0.5 보다 키우면 두 조가 겹쳐 딛어 안정적이지만 느려진다."""

    step_height_m: float = 0.035
    """나는 발이 들리는 최대 높이 [m]."""

    body_height_m: float = 0.12
    """base_link 가 지면에서 떠 있는 높이 [m].

    조인트를 전부 0 으로 두면 0.1267 m 가 된다(URDF 에서 계산). 조금 낮춰 두면
    무릎에 여유가 생겨 지면 요철을 흡수한다."""

    stance_radius_m: float = 0.126
    """중립 접지점이 `leg_center` 에서 떨어진 수평거리 [m].

    URDF 0 자세의 0.1262 m 를 그대로 쓴다. 넓힐수록 지지 다각형이 커져 안정적이지만
    다리가 더 뻗어 작업공간 여유가 준다 — 0.135 로 키우면 횡이동 명령에서 여유가
    3 mm 까지 줄어 IK 가 포화 직전이 된다(`test_gait.py` 가 그 여유를 잰다)."""

    max_v_mps: float = 0.12
    """전/후·좌/우 속도 상한 [m/s]. 보폭이 다리 길이를 넘지 않게 묶는다."""

    max_w_radps: float = 0.8
    """제자리 회전 각속도 상한 [rad/s]."""

    max_stride_m: float = 0.07
    """한 걸음의 최대 보폭 [m]. 속도가 빨라도 이보다 크게 내딛지 않는다.
    이걸 안 걸면 고속 명령에서 발이 작업공간 밖으로 나가 IK 가 포화된다."""

    # --- 운동학 모드에서 몸통이 흔들리는 정도 (--walk_mode kinematic 전용) ---
    body_pitch_deg: float = 3.5
    """보행이 만드는 몸체 pitch 진폭 [deg]. **합성값이다** — 물리 접지에서 나온 게
    아니라 보행 위상에 물려 넣은 모델이다. `--walk_mode physics` 에서는 안 쓴다."""

    body_roll_deg: float = 2.5
    """같은 의미의 roll 진폭 [deg]. pitch 와 90도 어긋나게 넣어, 삼각보행 특유의
    '몸이 8자로 흔들리는' 움직임이 난다."""

    body_bob_m: float = 0.006
    """몸이 아래위로 출렁이는 진폭 [m]. 조가 바뀔 때마다 한 번씩이라 **보행
    주파수의 2배**로 넣는다."""


@dataclass
class LegKinematics:
    """다리 하나의 평면 2링크 모델. 전부 URDF 에서 계산한 값이다."""

    leg: str
    center: np.ndarray
    """base_link 기준 `leg_center_XX` 위치 (3,) [m]."""
    mount_yaw: float
    """다리가 몸통에서 뻗어 나가는 방향 [rad]. coxa 조인트 rpy 의 yaw."""
    l1: float
    """leg_center -> femur 조인트 수평거리 [m]."""
    z0: float
    """leg_center 기준 femur 조인트 높이 [m]."""
    l2: float
    a2: float
    """femur 링크 길이와 오프셋 각 [m, rad]."""
    l3: float
    a3: float
    """tibia 링크 길이와 오프셋 각 [m, rad]."""
    y_off: float
    """발끝이 다리 평면에서 횡으로 비켜 있는 거리 [m]. 조인트각과 무관한 상수다."""

    def ik(self, target: np.ndarray) -> tuple[float, float, float]:
        """base_link 기준 발 목표점 -> (coxa, femur, tibla) 각도 [rad].

        닿을 수 없는 점은 **가장 가까운 닿는 점으로 눌러서** 각도를 돌려준다
        (예외를 던지지 않는다). 보행 중에 한두 프레임 작업공간을 벗어나도 다리가
        튀지 않고 뻗은 자세로 포화되는 편이 낫다.
        """
        # 1) leg_center 기준으로 옮기고, 장착 방향만큼 되돌려 다리 자신의 프레임으로.
        q = np.asarray(target, dtype=float) - self.center
        c, s = math.cos(-self.mount_yaw), math.sin(-self.mount_yaw)
        xl = c * q[0] - s * q[1]
        yl = s * q[0] + c * q[1]
        zl = q[2]

        # 2) coxa 는 수평면 안에서 다리를 겨눈다. 발끝이 평면에서 y_off 만큼 비켜
        #    있으므로, 목표점까지의 수평거리에서 그만큼 빼야 평면 안의 반경이 나온다
        #    (rho^2 = r^2 + y_off^2 이고, 이 관계는 theta1 과 무관하다).
        rho = math.hypot(xl, yl)
        r = math.sqrt(max(rho * rho - self.y_off * self.y_off, 0.0))
        theta1 = math.atan2(yl, xl) - math.atan2(self.y_off, r)

        # 3) 남은 둘은 반경-수직 평면 안의 2링크 문제다.
        a = r - self.l1
        b = zl - self.z0
        d = math.hypot(a, b)
        reach = self.l2 + self.l3
        if d > reach - 1e-6:  # 뻗어도 안 닿는다 -> 그 방향으로 최대한 뻗는다.
            d = reach - 1e-6
            scale = d / max(math.hypot(a, b), 1e-9)
            a, b = a * scale, b * scale

        cos_knee = (d * d - self.l2 * self.l2 - self.l3 * self.l3) / (2.0 * self.l2 * self.l3)
        # 0 자세의 무릎 각이 음수다(무릎이 뒤로 접힌다). 같은 가지를 계속 쓴다.
        knee = -math.acos(min(1.0, max(-1.0, cos_knee)))

        phi2 = math.atan2(b, a) - math.atan2(
            self.l3 * math.sin(knee), self.l2 + self.l3 * math.cos(knee)
        )
        theta2 = phi2 - self.a2
        theta3 = knee + self.a2 - self.a3
        return theta1, theta2, theta3

    def fk(self, theta1: float, theta2: float, theta3: float) -> np.ndarray:
        """위 모델의 정기구학. `ik` 를 검산할 때 쓴다 (`test_gait.py`)."""
        phi2 = theta2 + self.a2
        phi3 = theta2 + theta3 + self.a3
        r = self.l1 + self.l2 * math.cos(phi2) + self.l3 * math.cos(phi3)
        z = self.z0 + self.l2 * math.sin(phi2) + self.l3 * math.sin(phi3)
        ang = self.mount_yaw + theta1
        c, s = math.cos(ang), math.sin(ang)
        return self.center + np.array([r * c - self.y_off * s, r * s + self.y_off * c, z])


def _build_leg(leg: str, joints: dict) -> LegKinematics:
    """URDF 체인을 풀어 다리 하나의 상수를 뽑는다."""
    center, _ = link_pose(f"leg_center_{leg}", joints, {}, base="base_link")
    mount_yaw = float(joints[f"coxa_{leg}"]["rpy"][2])

    # 다리 자신의 프레임(장착 방향을 되돌린 축)에서 세 점을 본다.
    c, s = math.cos(-mount_yaw), math.sin(-mount_yaw)

    def in_leg_frame(link: str) -> tuple[float, float, float]:
        p, _ = link_pose(link, joints, {}, base=f"leg_center_{leg}")
        return c * p[0] - s * p[1], s * p[0] + c * p[1], float(p[2])  # (반경, 횡, 높이)

    r_femur, _, z_femur = in_leg_frame(f"femur_{leg}")
    r_tibla, _, z_tibla = in_leg_frame(f"tibla_{leg}")
    r_end, y_end, z_end = in_leg_frame(f"end_{leg}")

    dr2, dz2 = r_tibla - r_femur, z_tibla - z_femur
    dr3, dz3 = r_end - r_tibla, z_end - z_tibla
    return LegKinematics(
        leg=leg,
        center=center,
        mount_yaw=mount_yaw,
        l1=r_femur,
        z0=z_femur,
        l2=math.hypot(dr2, dz2),
        a2=math.atan2(dz2, dr2),
        l3=math.hypot(dr3, dz3),
        a3=math.atan2(dz3, dr3),
        y_off=y_end,
    )


class HexapodGait:
    """삼각보행 생성기.

    Example:
        gait = HexapodGait("assets/rospider_description/rospider.urdf")
        targets = gait.joint_targets(phase=0.25, cmd=(0.1, 0.0, 0.0))
        # -> {"coxa_LF_joint": 0.03, "femur_LF_joint": -0.12, ...} 18개
    """

    def __init__(self, urdf_path: str | Path, params: GaitParams | None = None) -> None:
        self.params = params or GaitParams()
        joints = load_joints(Path(urdf_path))
        self.legs: dict[str, LegKinematics] = {leg: _build_leg(leg, joints) for leg in LEGS}

    # ------------------------------------------------------------------ 중립 자세
    def neutral_foot(self, leg: str) -> np.ndarray:
        """명령이 0 일 때 그 다리가 딛는 점 (base_link 기준, 3,) [m]."""
        k = self.legs[leg]
        r = self.params.stance_radius_m
        return np.array(
            [
                k.center[0] + r * math.cos(k.mount_yaw),
                k.center[1] + r * math.sin(k.mount_yaw),
                -self.params.body_height_m,
            ]
        )

    def stand_joint_targets(self) -> dict[str, float]:
        """가만히 선 자세. 시작할 때와 명령이 0 일 때 쓴다."""
        return self.joint_targets(phase=0.0, cmd=(0.0, 0.0, 0.0))

    # ------------------------------------------------------------------ 보행
    def clip_cmd(self, cmd) -> np.ndarray:
        """(vx, vy, wz) 를 상한 안으로 누른다."""
        p = self.params
        vx, vy, wz = (float(v) for v in cmd)
        speed = math.hypot(vx, vy)
        if speed > p.max_v_mps:
            vx, vy = vx * p.max_v_mps / speed, vy * p.max_v_mps / speed
        wz = min(p.max_w_radps, max(-p.max_w_radps, wz))
        return np.array([vx, vy, wz])

    def half_stride(self, leg: str, cmd) -> np.ndarray:
        """그 다리가 중립점에서 앞뒤로 오가는 **절반 보폭** (x, y) [m].

        회전 명령은 다리마다 다른 방향으로 작용한다 — 제자리 회전에서 좌측 다리와
        우측 다리는 서로 반대로 쓸려야 하기 때문이다. 그래서 중립점의 위치가 식에
        들어간다 (``wz x n``).
        """
        vx, vy, wz = self.clip_cmd(cmd)
        n = self.neutral_foot(leg)
        s = np.array([vx - wz * n[1], vy + wz * n[0]])
        half = s * (self.params.period_s * self.params.duty) / 2.0
        length = float(np.linalg.norm(half))
        limit = self.params.max_stride_m / 2.0
        if length > limit:
            half = half * (limit / length)
        return half

    def foot_target(self, leg: str, phase: float, cmd) -> np.ndarray:
        """위상과 명령에서 그 다리의 발 목표점 (base_link 기준, 3,) [m].

        Args:
            leg: "LF" 등.
            phase: 보행 위상 [0, 1). 두 조가 0.5 만큼 어긋나게 쓴다.
            cmd: (vx, vy, wz).
        """
        p = self.params
        local = phase if leg in TRIPOD_A else (phase + 0.5) % 1.0
        half = self.half_stride(leg, cmd)
        foot = self.neutral_foot(leg).copy()

        if local < p.duty:
            # 딛는 중 — 앞(+half)에서 뒤(−half)로 일정하게 쓸린다. 그만큼 몸이 나간다.
            u = local / p.duty
            foot[:2] += half * (1.0 - 2.0 * u)
        else:
            # 나는 중 — 뒤에서 앞으로 돌아오면서 반주기 동안 포물선으로 들린다.
            u = (local - p.duty) / (1.0 - p.duty)
            foot[:2] += half * (-1.0 + 2.0 * u)
            # 발 드는 높이는 보폭에 비례한다. 명령이 0 이면 0 이 되어 **제자리걸음을
            # 하지 않는다.** 이걸 안 걸면 서 있으란 명령에도 다리가 계속 들려
            # 로봇이 부르르 떨고, 물리 모드에서는 그 진동이 IMU 에 그대로 들어온다.
            foot[2] += p.step_height_m * math.sin(math.pi * u) * self._lift_scale(half)
        return foot

    def _lift_scale(self, half: np.ndarray) -> float:
        """보폭이 작으면 발도 덜 든다. 0 ~ 1."""
        full = self.params.max_stride_m / 2.0 * 0.25  # 최대 보폭의 1/4 이면 다 든다
        return min(1.0, float(np.linalg.norm(half)) / max(full, 1e-9))

    def joint_targets(self, phase: float, cmd) -> dict[str, float]:
        """18개 다리 조인트의 목표각 [rad]. 이름은 URDF 그대로다."""
        out: dict[str, float] = {}
        for leg in LEGS:
            t1, t2, t3 = self.legs[leg].ik(self.foot_target(leg, phase, cmd))
            out[joint_name("coxa", leg)] = t1
            out[joint_name("femur", leg)] = t2
            out[joint_name("tibla", leg)] = t3
        return out

    def swinging(self, phase: float) -> tuple[str, ...]:
        """지금 공중에 있는 다리들. 패널과 로그에 쓴다."""
        return TRIPOD_B if phase < self.params.duty else TRIPOD_A

    # ------------------------------------------------- 운동학 모드용 몸통 흔들림
    def body_sway(self, phase: float, cmd) -> tuple[float, float, float]:
        """보행이 만드는 몸통 흔들림 (dz [m], pitch [rad], roll [rad]).

        **합성 모델이다.** 물리 접지에서 나온 값이 아니라, 보행 위상에 물려 넣은
        사인파다. `--walk_mode kinematic` 에서 루트를 직접 쓸 때만 쓴다
        (`physics` 모드에서는 PhysX 가 만드는 진짜 흔들림을 IMU 가 읽는다).

        진폭은 명령 속도에 비례한다 — 제자리에 서 있으면 흔들리지 않는다.
        """
        p = self.params
        vx, vy, wz = self.clip_cmd(cmd)
        effort = min(
            1.0,
            math.hypot(vx, vy) / max(p.max_v_mps, 1e-9) + abs(wz) / max(p.max_w_radps, 1e-9),
        )
        ang = 2.0 * math.pi * phase
        pitch = math.radians(p.body_pitch_deg) * math.sin(ang) * effort
        roll = math.radians(p.body_roll_deg) * math.cos(ang) * effort
        # 조가 바뀔 때마다 한 번 가라앉는다 -> 보행 주파수의 2배.
        dz = -p.body_bob_m * (1.0 - math.cos(2.0 * ang)) / 2.0 * effort
        return dz, pitch, roll


def advance_phase(phase: float, dt: float, period_s: float) -> float:
    """위상을 dt 만큼 진행시킨다. [0, 1) 로 감는다."""
    return (phase + dt / period_s) % 1.0


if __name__ == "__main__":  # 빠른 눈 확인용. 정식 검증은 test_gait.py 가 한다.
    import argparse

    ap = argparse.ArgumentParser(description="보행 생성기 상수와 한 주기를 찍어 본다")
    ap.add_argument("--urdf", default="assets/rospider_description/rospider.urdf")
    args = ap.parse_args()

    gait = HexapodGait(args.urdf)
    print("URDF 에서 뽑은 다리 상수\n")
    print(f"{'leg':5s}{'mount_yaw[deg]':>16s}{'L1':>9s}{'L2':>9s}{'L3':>9s}{'a2[deg]':>10s}{'a3[deg]':>10s}")
    for leg in LEGS:
        k = gait.legs[leg]
        print(
            f"{leg:5s}{math.degrees(k.mount_yaw):16.2f}{k.l1:9.5f}{k.l2:9.5f}{k.l3:9.5f}"
            f"{math.degrees(k.a2):10.2f}{math.degrees(k.a3):10.2f}"
        )

    print("\n전진 0.1 m/s 한 주기에서 LF 발이 그리는 자취 (base_link 기준)\n")
    for i in range(9):
        ph = i / 8.0 % 1.0
        f = gait.foot_target("LF", ph, (0.1, 0.0, 0.0))
        state = "swing" if "LF" in gait.swinging(ph) else "stance"
        print(f"  phase={ph:.3f} {state:7s} foot=({f[0]:+.4f}, {f[1]:+.4f}, {f[2]:+.4f})")
