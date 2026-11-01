"""보행 학습용 RosPider ArticulationCfg. **Kit 이 뜬 뒤에 import 해야 한다.**

sensor_fusion/rospider_cfg.py 와 다른 점
----------------------------------------
- **베이스가 떠 있는 USD** 를 쓴다 (`02_urdf_to_usd.py --no-fix-base` -> assets/usd/rospider_float.usd).
  인식 과제는 베이스를 월드에 고정했지만, 보행 학습은 몸이 실제로 움직여야 한다.
- **접촉 센서를 켠다** (`activate_contact_sensors=True`). 발 착지 시간, 몸통 충돌 종료에 쓴다.
- 서 있는 자세를 gait.py 의 IK 로 계산해 초기 관절각으로 쓴다 (몸통 높이 0.12 m, 발 반경 0.126 m).
  URDF 를 못 찾으면 아래 STAND_FALLBACK (2026-11-01 같은 계산으로 뽑아 둔 값) 을 쓴다.

바디 이름 (merge_fixed_joints=True 변환 기준, 02_urdf_to_usd.py 기본값)
- 루트: base_footprint (base_link 가 여기 흡수됨). 그래서 루트는 지면 높이에 놓는다.
- 발: end_* 가 tibla_* 에 흡수됨 -> 발 접촉은 ``tibla_.*`` 바디로 본다.

[미검증] 다리 PD 게인(stiffness 20, damping 1)과 토크 3.4 N*m 는 sensor_fusion 과 같은 출발점이다.
실기 서보 응답에 맞춘 값이 아니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

REPO = Path(__file__).resolve().parents[2]
DEFAULT_USD = REPO / "assets" / "usd" / "rospider_float.usd"
DEFAULT_URDF = REPO / "assets" / "rospider_description" / "rospider.urdf"

LEG_JOINTS = ["coxa_.*_joint", "femur_.*_joint", "tibla_.*_joint"]
FOOT_BODIES = "tibla_.*"
BASE_BODY = "base_footprint"

ARM_FOLD = {"joint1": 0.0, "joint2": 0.85, "joint3": -1.60, "joint4": -1.26, "joint5": 0.0}
"""팔 자세. sensor_fusion 의 ARM_OBSERVE_POSE 와 같다(카메라가 전방 24도 아래를 본다).
걷는 동안 팔은 이 자세로 붙잡아 둔다 — 정책은 다리 18개만 움직인다."""

STAND_FALLBACK = {
    "coxa_LF_joint": -0.0119, "femur_LF_joint": 0.0950, "tibla_LF_joint": -0.1186,
    "coxa_LM_joint": -0.0138, "femur_LM_joint": 0.0960, "tibla_LM_joint": -0.1187,
    "coxa_LR_joint": -0.0236, "femur_LR_joint": 0.0948, "tibla_LR_joint": -0.1173,
    "coxa_RF_joint": 0.0118, "femur_RF_joint": 0.0961, "tibla_RF_joint": -0.1188,
    "coxa_RM_joint": 0.1385, "femur_RM_joint": 0.0364, "tibla_RM_joint": -0.0723,
    "coxa_RR_joint": 0.0236, "femur_RR_joint": 0.0948, "tibla_RR_joint": -0.1184,
}
"""gait.HexapodGait(URDF).stand_joint_targets() 결과 (rad). RM 이 다른 건 URDF 의 end_RM 오프셋 때문(gait.py 참고)."""


def stand_pose(urdf_path: Path = DEFAULT_URDF) -> dict[str, float]:
    if urdf_path.exists():
        sys.path.insert(0, str(REPO / "tasks" / "sensor_fusion"))
        try:
            import gait  # noqa: PLC0415

            return {k: float(v) for k, v in gait.HexapodGait(urdf_path).stand_joint_targets().items()}
        except Exception as e:  # noqa: BLE001
            print(f"[robot_cfg] gait 로 선 자세를 못 구해 STAND_FALLBACK 을 쓴다: {e}")
    return dict(STAND_FALLBACK)


def make_robot_cfg(usd_path: Path = DEFAULT_USD, urdf_path: Path = DEFAULT_URDF) -> ArticulationCfg:
    if not Path(usd_path).exists():
        print(f"[robot_cfg] [경고] {usd_path} 가 없다. 먼저:\n"
              "    python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base")
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 루트(base_footprint)는 바닥 투영점. 선 자세에서 base_link 가 0.12 m 라 루트는 약 4 mm.
            # 조금 띄워서 떨어뜨린다.
            pos=(0.0, 0.0, 0.03),
            joint_pos={**stand_pose(urdf_path), **ARM_FOLD, "[rl]_.*joint": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=LEG_JOINTS,
                effort_limit_sim=3.4,       # 실기 서보 35 kg*cm (sensor_fusion 과 같은 값)
                velocity_limit_sim=6.0,
                stiffness=20.0,             # [미검증] 출발점
                damping=1.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-5]"],
                effort_limit_sim=10.0,
                velocity_limit_sim=3.0,
                stiffness=200.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["[rl]_.*joint"],
                effort_limit_sim=2.0,
                velocity_limit_sim=3.0,
                stiffness=20.0,
                damping=2.0,
            ),
        },
    )
