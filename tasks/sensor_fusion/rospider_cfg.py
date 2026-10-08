"""RosPider 로봇/센서/씬 설정. **Kit 이 뜬 뒤에 import 해야 한다.**

여기 적힌 숫자는 전부 RosPider URDF(Hiwonder/ROSpider 의 rospider_description) 에서
`urdf_fk.py` 로 뽑은 값이다. 추측값이 아니다. 다만 "서 있는 다리 자세"와
"PD 게인"은 실기에서 맞춰본 값이 아니라 출발점이다(아래 주석에 표시).

센서를 어디에 붙였나
--------------------
실기의 뎁스 카메라는 **팔 끝(link4)** 에 달려 있다. 그대로 쓰면 카메라 자세가 팔
조인트에 따라 바뀌어서, 융합에 필요한 `R_cam<-imu` 가 상수가 아니게 된다. 그래서:

- 팔은 **관측 자세로 고정**한다 (joint2=0.85, joint3=-1.60, joint4=-1.26).
  이 자세에서 카메라는 전방을 보며 약 24.2도 아래로 숙는다.
- 카메라 prim 은 그 자세에서 계산된 `base_link` 기준 고정 오프셋으로 붙인다.
  겉보기(팔 끝에 달린 모양)와 실제 센서 위치가 일치한다.
- IMU 도 `base_link` 에 붙인다. URDF 의 imu_joint 오프셋 그대로다.

덕분에 `R_cam<-imu` 가 상수가 되고, 그게 바로 fusion.py 의 기본값이다.
팔을 다른 자세로 쓰고 싶으면 `urdf_fk.py --joints ...` 로 값을 다시 뽑아
`FusionParams.r_cam_imu` 와 아래 `CAM_*` 상수를 같이 갱신하면 된다.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------- URDF 에서 뽑은 상수
BASE_HEIGHT = 0.11607
"""base_footprint -> base_link. URDF 의 ``0.0965/2 + 0.0678410821576746``."""

ARM_OBSERVE_POSE = {"joint1": 0.0, "joint2": 0.85, "joint3": -1.60, "joint4": -1.26, "joint5": 0.0}
"""카메라가 전방 약 24.2도 하향을 보는 팔 자세. 다리 제어와 무관하게 고정한다."""

CAM_POS_IN_BASE = (0.13125, 0.0013, 0.25086)
"""위 자세에서 depth_cam_frame 의 base_link 기준 위치 [m]."""

CAM_QUAT_IN_BASE_WORLD_CONV = (0.97753, 0.0, 0.210794, 0.0)
"""같은 자세에서의 회전, **world 규약**(전방 +X, 상방 +Z) 쿼터니언 (w,x,y,z).

`CameraCfg.OffsetCfg(convention="world")` 에 그대로 넣는다. +Y 축 기준 24.3도 회전
= 전방 축이 그만큼 아래를 향한다는 뜻이라 사람이 읽기 쉽다. ROS 광학 규약
(convention="ros") 으로 넣고 싶으면 (-0.38422, 0.593616, -0.593607, 0.384222) 이다.
"""

IMU_POS_IN_BASE = (0.0048416, 0.011168, -0.0057398)
"""URDF imu_joint 의 origin xyz."""

IMU_QUAT_IN_BASE = (0.70710678, 0.0, 0.0, -0.70710678)
"""URDF imu_joint 의 origin rpy (0, 0, -pi/2) 를 쿼터니언 (w,x,y,z) 로."""

CAM_HEIGHT_AT_LEVEL = BASE_HEIGHT + CAM_POS_IN_BASE[2]
"""수평 자세에서 바닥 -> 카메라 광학 중심 높이 [m]. 0.367 m."""


def pitch_to_quat(pitch_deg: float) -> tuple[float, float, float, float]:
    """+Y 축 회전(코를 숙이는 방향)을 쿼터니언 (w,x,y,z) 로. 양수면 앞으로 숙인다."""
    half = math.radians(pitch_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def make_robot_cfg(usd_path: str, pitch_deg: float = 0.0) -> ArticulationCfg:
    """RosPider ArticulationCfg.

    Args:
        usd_path: 02_urdf_to_usd.py 가 만든 USD 경로.
        pitch_deg: 몸체를 앞으로 숙이는 각도 [deg]. 실험 조건을 바꾸는 손잡이다.
            USD 를 ``fix_base=True`` 로 변환했으므로 이 자세가 그대로 유지된다.
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT),
            rot=pitch_to_quat(pitch_deg),
            # [미검증] 다리 각도는 실기/GUI 에서 맞춰본 값이 아니다. 0 으로 두면 다리가
            # 쭉 뻗은 자세가 되어 메시가 바닥을 뚫을 수 있다. GUI(--view) 로 보면서
            # 세 조인트 부호를 찾아 넣자. base 가 고정이라 넘어지지는 않는다.
            joint_pos={
                "coxa_.*_joint": 0.0,
                "femur_.*_joint": 0.0,
                "tibla_.*_joint": 0.0,
                **ARM_OBSERVE_POSE,
                "[rl]_.*joint": 0.0,
            },
        ),
        actuators={
            # 다리: 실기 서보는 35 kg*cm = 약 3.4 N*m. URDF 에 적힌 effort=1 은 과소값이라
            # 여기서 덮는다. 게인은 [미검증] 출발점이다.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["coxa_.*_joint", "femur_.*_joint", "tibla_.*_joint"],
                effort_limit_sim=3.4,
                velocity_limit_sim=6.0,
                stiffness=20.0,
                damping=1.0,
            ),
            # 팔: 관측 자세를 흔들림 없이 붙잡아야 카메라 외부 파라미터가 유지된다.
            # 그래서 다리보다 훨씬 단단하게 잡는다.
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


def make_camera_cfg(width: int = 160, height: int = 120, with_rgb: bool = True) -> CameraCfg:
    """뎁스 카메라.

    - ``distance_to_image_plane`` 이 뎁스다(단위 m, 광학 z축 기준). ``"depth"`` 는
      같은 것의 별칭이다. ``CameraCfg.data_types`` 기본값은 ``["rgb"]`` 라서
      **뎁스를 쓰려면 반드시 명시해야 한다.**
    - 실기 뎁스 카메라는 640x480 급이지만 VRAM 8 GB 에서는 해상도를 낮춰야 한다.
      160x120 이면 융합 판정에 충분하고, 노션 노트용 이미지로도 읽을 만하다.
    - 카메라를 켰으면 **실행할 때 `--enable_cameras` 가 필요하다**(헤드리스에서도).
    """
    data_types = ["distance_to_image_plane"]
    if with_rgb:
        # 노션 노트에 "센서별 화면" 을 싣기 위해 RGB 도 같이 받는다.
        # VRAM 이 빠듯하면 이것부터 끄자.
        data_types.insert(0, "rgb")

    # 수평 화각 70도: 2*atan(20.955 / (2*14.96)) = 70.0도
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/depth_cam",
        update_period=0.0,  # 매 물리 스텝 갱신 -> IMU 와 측정 시점이 어긋나지 않는다.
        width=width,
        height=height,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.96,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAM_POS_IN_BASE,
            rot=CAM_QUAT_IN_BASE_WORLD_CONV,
            convention="world",
        ),
        # 먼 곳/하늘은 inf 로 들어온다. fusion.py 가 isfinite 로 걸러낸다.
        depth_clipping_behavior="none",
    )


def make_imu_cfg() -> ImuCfg:
    """IMU.

    `gravity_bias=(0,0,0)` 으로 둔 이유: 기본값 (0,0,9.81) 은 실제 IMU 처럼 정지
    상태에서도 +9.81 이 읽히게 만든다. 그러면 충격 판정 임계값을 '9.81 근처에서
    얼마나 벗어났나' 로 복잡하게 잡아야 한다. 0 으로 두면 `lin_acc_b` 가 순수
    운동가속이 되어 `|a| > 6 m/s^2` 같은 단순 임계값이 바로 먹는다.
    자세(기울기) 는 `projected_gravity_b` 에서 얻고, 이 값은 gravity_bias 와 무관하다.
    """
    return ImuCfg(
        # merge_fixed_joints=True 로 변환하면 imu_link 가 base_link 에 흡수돼
        # prim 이 사라진다. 그래서 base_link 에 붙이고 URDF 오프셋을 직접 준다.
        # (--no-merge-fixed-joints 로 변환했다면 prim_path 를 ".../imu_link" 로,
        #  offset 을 기본값으로 두면 된다.)
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(pos=IMU_POS_IN_BASE, rot=IMU_QUAT_IN_BASE),
        gravity_bias=(0.0, 0.0, 0.0),
        update_period=0.0,
        debug_vis=False,
    )


@configclass
class FusionSceneCfg(InteractiveSceneCfg):
    """바닥 + 조명 + 로봇 + 장애물. 장애물은 정적 프림이라 굴러가지 않는다."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)),
    )
    robot: ArticulationCfg = None  # build_scene_cfg 에서 채운다
    camera: CameraCfg = None
    imu: ImuCfg = None
    obstacle: RigidObjectCfg = None
    backdrop: AssetBaseCfg = None


def build_scene_cfg(
    usd_path: str,
    pitch_deg: float = 0.0,
    obstacle_dist: float = 0.8,
    obstacle_height: float = 0.15,
    obstacle_width: float = 0.24,
    cam_width: int = 160,
    cam_height: int = 120,
    with_rgb: bool = True,
    num_envs: int = 1,
) -> FusionSceneCfg:
    """실험 조건 하나를 씬 설정으로 만든다.

    Args:
        obstacle_dist: 카메라 기준 전방 수평거리 [m]. 카메라가 base 보다 앞에 있으므로
            base 기준 x 는 여기에 카메라 오프셋을 더한 값이 된다.
        obstacle_height: 장애물 높이 [m]. 0.02 처럼 낮게 주면 '넘어갈 수 있는 단차' 가
            되고, 융합은 이걸 장애물로 세지 않아야 한다.
    """
    cfg = FusionSceneCfg(num_envs=num_envs, env_spacing=4.0)
    cfg.robot = make_robot_cfg(usd_path, pitch_deg=pitch_deg)
    cfg.camera = make_camera_cfg(cam_width, cam_height, with_rgb)
    cfg.imu = make_imu_cfg()

    # 카메라의 월드 x 위치. 몸체를 숙이면 카메라가 앞으로 나온다.
    pitch = math.radians(pitch_deg)
    cam_x = math.cos(pitch) * CAM_POS_IN_BASE[0] + math.sin(pitch) * CAM_POS_IN_BASE[2]
    obstacle_x = cam_x + obstacle_dist + 0.1  # +0.1 은 상자 중심까지 (깊이 0.2 의 절반)

    # 장애물은 RigidObject + kinematic 이다. 중력에 안 떨어지고, 그러면서도
    # write_root_pose_to_sim 으로 **실행 중에 위치를 옮길 수 있다.** 거리 조건을
    # 바꿀 때마다 Kit 을 다시 띄우지 않아도 되는 이유다.
    cfg.obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, obstacle_width, max(obstacle_height, 0.001)),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(obstacle_x, 0.0, obstacle_height / 2)),
    )
    # 뒷벽. 뎁스가 전부 inf 인 화면은 보기에도 헷갈리고, 바닥 추정 분포도 왜곡된다.
    cfg.backdrop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/backdrop",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 2.0, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.62)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cam_x + 1.8, 0.0, 0.3)),
    )
    return cfg
