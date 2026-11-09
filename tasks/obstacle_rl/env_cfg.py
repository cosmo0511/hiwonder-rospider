"""장애물 넘기 강화학습 환경 설정 (관측 / 행동 / 명령 / 보상 / 종료 / 이벤트). **Kit 이 뜬 뒤에 import.**

Isaac Lab 기본 보행 과제(``LocomotionVelocityRoughEnvCfg``: ANYmal, Go2 등이 쓰는 것)를 물려받아
로봇, 지형, 파이프만 RosPider 와 실물 코스에 맞게 바꾼다. 처음부터 다 짜는 것보다 검증된
틀을 쓰는 쪽이 덜 틀린다.

무엇을 하는 정책인가
---------------------
"전진 속도 명령(0.04 ~ 0.12 m/s)을 따라 걸으면서 앞의 장애물을 넘는다." 장애물 위치는
모른다. 대신 몸 아래 높이 스캔(0.8 x 0.6 m, 4 cm 간격)을 본다.

| 구분 | 내용 |
|---|---|
| 관측 (정책) | 몸 선속도/각속도, 중력 방향, 속도 명령, 다리 관절각/속도, 직전 행동, 높이 스캔 |
| 행동 | 다리 18개 관절 목표각 = 선 자세 + 0.25 x 출력 (rad). 팔과 그리퍼는 고정 |
| 명령 | 전진 0.04 ~ 0.12 m/s, 옆 +-0.02, 방향 유지(heading +-0.15 rad) |
| 보상 | 속도 추종, 전진 진척, 발 들기 시간 / 수직 흔들림, 토크, 행동 변화, 허벅지 충돌, 관절 한계 벌점 |
| 종료 | 시간 초과(40 s), 몸통이 땅/장애물에 닿음, 1 rad 넘게 기울어짐 |
| 이벤트 | 마찰, 몸통 질량 +0.8 ~ 1.8 kg (URDF 1.66 kg vs 실물 3.2 kg), 리셋 자세, 약한 밀기, 파이프 배치 |

[미검증 / 출발점] 보상 가중치와 명령 범위는 RosPider 에 맞춰 줄인 출발값이다. 학습을 돌려 보고 조정한다.
높이 스캔은 시뮬에서만 얻는 특권 관측이다 — 실기에는 그런 센서가 없다. 실기로 옮기려면
나중에 뎁스 카메라 관측으로 바꾸거나(정책 증류) 스캔을 뎁스에서 만들어야 한다.
"""

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
)

from . import course
from . import mdp as obs_mdp
from .robot_cfg import BASE_BODY, FOOT_BODIES, LEG_JOINTS, make_robot_cfg
from .terrain_cfg import OBSTACLE_TERRAINS_CFG, REAL_COURSE_TERRAINS_CFG


def _pipe_cfg(i: int) -> RigidObjectCfg:
    """굴러가는 강관 Ø48.6 x 1.0 m. 축은 y(코스 가로). 시작 자리는 reset_pipes 가 정한다.

    [주의] 관성은 Isaac Lab 이 꽉 찬 원통으로 계산한다(m r^2 / 2). 실제 관(m r^2 에 가까움)보다
    굴리기 쉽다. 정확히 맞추려면 USD 에서 관성을 직접 넣어야 한다.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Pipe{i}",
        spawn=sim_utils.CylinderCfg(
            radius=course.PIPE_D / 2,
            height=course.PIPE_LEN,
            axis="Y",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=course.PIPE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.65, 0.68), metallic=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(course.PIPE_PARK[i], 0.0, course.PIPE_D / 2)),
    )


@configclass
class ObstacleSceneCfg(MySceneCfg):
    pipe_0: RigidObjectCfg = _pipe_cfg(0)
    pipe_1: RigidObjectCfg = _pipe_cfg(1)


@configclass
class RoSpiderObstacleEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: ObstacleSceneCfg = ObstacleSceneCfg(num_envs=64, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # ---------------- 씬
        self.scene.robot = make_robot_cfg()
        self.scene.terrain.terrain_generator = copy.deepcopy(OBSTACLE_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + BASE_BODY
        self.scene.height_scanner.pattern_cfg.resolution = 0.04
        self.scene.height_scanner.pattern_cfg.size = [0.8, 0.6]
        self.scene.height_scanner.offset.pos = (0.15, 0.0, 20.0)   # 몸보다 15 cm 앞을 중심으로
        self.episode_length_s = 40.0   # 출발점 ~ 작업대 약 2.4 m 를 0.1 m/s 로

        # ---------------- 행동: 다리 18개만
        self.actions.joint_pos.joint_names = LEG_JOINTS
        self.actions.joint_pos.scale = 0.25

        # ---------------- 관측: 관절은 다리만
        # 항마다 따로 만든다. 한 객체를 공유하면 첫 항이 resolve 하며 joint_ids 를 채워 두 번째 항에서 불일치 에러가 난다.
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}

        # ---------------- 명령: 앞으로만, 느리게 (gait.py 최대 0.12 m/s)
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.04, 0.12)
        cmd.ranges.lin_vel_y = (-0.02, 0.02)
        cmd.ranges.ang_vel_z = (-0.3, 0.3)
        cmd.ranges.heading = (-0.15, 0.15)
        cmd.heading_command = True
        cmd.rel_standing_envs = 0.05
        cmd.resampling_time_range = (10.0, 10.0)

        # ---------------- 이벤트
        ev = self.events
        ev.physics_material.params["static_friction_range"] = (0.5, 1.0)
        ev.physics_material.params["dynamic_friction_range"] = (0.4, 0.9)
        ev.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_BODY)
        ev.add_base_mass.params["mass_distribution_params"] = (0.8, 1.8)   # 실물 3.2 kg 에 가깝게
        ev.add_base_mass.params["operation"] = "add"
        if getattr(ev, "base_com", None) is not None:
            ev.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_BODY)
            ev.base_com.params["com_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)}
        ev.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_BODY)
        ev.reset_base.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.15, 0.15), "yaw": (-0.15, 0.15)}
        ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        ev.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        ev.push_robot.params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
        ev.push_robot.interval_range_s = (10.0, 15.0)
        ev.reset_pipes = EventTerm(func=obs_mdp.reset_pipes, mode="reset", params={"jitter_x": 0.03})

        # ---------------- 보상 (작은 로봇, 느린 속도에 맞춘 출발값)
        rw = self.rewards
        rw.track_lin_vel_xy_exp.weight = 1.5
        rw.track_lin_vel_xy_exp.params["std"] = 0.05
        rw.track_ang_vel_z_exp.weight = 0.5
        rw.track_ang_vel_z_exp.params["std"] = 0.25
        rw.forward_progress = RewTerm(func=obs_mdp.forward_progress, weight=5.0,
                                      params={"command_name": "base_velocity"})
        rw.lin_vel_z_l2.weight = -0.5          # 계단/방지턱은 위아래로 움직여야 해서 기본(-2)보다 약하게
        rw.ang_vel_xy_l2.weight = -0.05
        rw.dof_torques_l2.weight = -2.0e-5
        rw.dof_acc_l2.weight = -2.5e-7
        rw.action_rate_l2.weight = -0.01
        rw.feet_air_time.weight = 0.2
        rw.feet_air_time.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=FOOT_BODIES)
        rw.feet_air_time.params["threshold"] = 0.3   # 삼각보행 주기 1.2 s 면 한 발이 약 0.6 s 떠 있다
        rw.undesired_contacts.weight = -0.5
        rw.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="femur_.*")
        rw.flat_orientation_l2.weight = -0.2
        rw.dof_pos_limits.weight = -1.0
        rw.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)

        # ---------------- 종료
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=BASE_BODY)
        self.terminations.bad_orientation = DoneTerm(func=loco_mdp.bad_orientation, params={"limit_angle": 1.0})

        # 파이프 배치가 '열 = 타일 종류' 규칙에 기대므로 지형 생성기는 항상 커리큘럼 모드로 둔다.
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class RoSpiderObstacleEnvCfg_PLAY(RoSpiderObstacleEnvCfg):
    """평가: 실물 맵 1, 맵 2 를 그대로. 무작위화와 관측 잡음을 끄고 0.1 m/s 직진만 명령한다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.terrain.terrain_generator = copy.deepcopy(REAL_COURSE_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        ev = self.events
        ev.physics_material.params["static_friction_range"] = (0.8, 0.8)
        ev.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        ev.add_base_mass.params["mass_distribution_params"] = (1.54, 1.54)   # 3.2 - 1.66
        ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        ev.push_robot = None
        ev.base_external_force_torque = None
        ev.reset_pipes.params["jitter_x"] = 0.0
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.1, 0.1)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        self.terminations.platform = DoneTerm(func=obs_mdp.reached_platform)
        self.scene.terrain.terrain_generator.curriculum = True



# ---------------------------------------------------------------- 단계별 학습용 옵션 (train.py 플래그)
# 기본 설정은 그대로 두고, 필요한 것만 덮어쓴다.

def make_flat(cfg: RoSpiderObstacleEnvCfg) -> None:
    """평지에서 보행만 먼저 배운다. 파이프는 출발 뒤쪽 주차 자리에 그대로 둔다."""
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.terrain_levels = None
    cfg.events.reset_pipes = None
    if getattr(cfg.terminations, "reached_goal", None) is not None:   # 코스 목표는 지형 생성기가 있어야 계산됨
        cfg.terminations.reached_goal = None
    if getattr(cfg.rewards, "goal_bonus", None) is not None:
        cfg.rewards.goal_bonus = None


def make_blind(cfg: RoSpiderObstacleEnvCfg) -> None:
    """정책(actor)은 실기에서 얻을 수 있는 값만 본다: IMU(각속도, 중력 방향), 명령, 다리 관절, 직전 행동.
    몸 선속도와 높이 스캔은 critic 만 본다(학습 때만 쓰는 특권 관측). 에이전트 쪽은 obs_groups 를 맞춰야 한다.
    """
    critic = copy.deepcopy(cfg.observations.policy)
    critic.enable_corruption = False
    cfg.observations.critic = critic
    cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.height_scan = None


def set_physics(cfg: RoSpiderObstacleEnvCfg, vel_iters=None, pos_iters=None, leg_armature=None,
                leg_vel_limit=None, dt=None) -> None:
    """물리 설정 덮어쓰기. None 이면 그대로."""
    rob = cfg.scene.robot
    if vel_iters is not None:
        rob.spawn.articulation_props.solver_velocity_iteration_count = vel_iters
    if pos_iters is not None:
        rob.spawn.articulation_props.solver_position_iteration_count = pos_iters
    if leg_armature is not None:
        rob.actuators["legs"].armature = leg_armature
    if leg_vel_limit is not None:
        rob.actuators["legs"].velocity_limit_sim = leg_vel_limit
    if dt is not None:
        ctrl = cfg.sim.dt * cfg.decimation
        cfg.sim.dt = dt
        cfg.decimation = round(ctrl / dt)
        cfg.sim.render_interval = cfg.decimation
        cfg.scene.contact_forces.update_period = dt


def physics_from_args(cfg: RoSpiderObstacleEnvCfg, args) -> None:
    set_physics(cfg, args.vel_iters, args.pos_iters, args.leg_armature, args.leg_vel_limit, args.dt)


def use_wasd_gains(cfg: RoSpiderObstacleEnvCfg) -> None:
    """다리 게인을 rospider_wasd.py(실제로 걷는 걸 확인한 값)에 맞춘다.
    그 스크립트는 degree 단위 드라이브: KP 1.0 /deg, KD 0.02 /(deg/s), 최대 토크 2.0 -> rad 단위로 바꾼다.
    """
    legs = cfg.scene.robot.actuators["legs"]
    legs.stiffness = math.degrees(1.0)     # 57.3 N*m/rad
    legs.damping = math.degrees(0.02)      # 1.146 N*m*s/rad
    legs.effort_limit_sim = 2.0


def use_tripod_gait(cfg: RoSpiderObstacleEnvCfg) -> None:
    """1단계: wasd tripod 걸음을 배우게 한다. make_blind 보다 **먼저** 불러야 critic 에도 박자가 들어간다.

    - 관측: 걸음 박자 sin/cos (1.2 Hz)
    - 보상: 기준 tripod 궤적 모방, 박자대로 접촉, 스윙 발 높이, 디딘 발 미끄러짐 벌점, 방향 유지 강화
    - 명령: 전진 0.08 ~ 0.20 m/s (wasd 걸음이 0.18 m/s). 항상 걷는다(서 있는 env 없음)
    - 행동 크기 0.25 -> 0.5 (wasd 는 허벅지를 0.6 rad 까지 든다)
    """
    cfg.observations.policy.gait_clock = ObsTerm(func=obs_mdp.gait_clock)
    cfg.actions.joint_pos.scale = 0.5
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (0.08, 0.20)
    cmd.rel_standing_envs = 0.0
    rw = cfg.rewards
    # 허용 폭 0.5 rad: 베이스라인 tripod 가 기준과 RMS 0.10 rad (점수 0.96). 0.15 로 했더니 정책 오차 0.89 rad 에서 점수가 0 에 붙어 안 배웠음
    rw.imitate_tripod = RewTerm(func=obs_mdp.imitate_tripod, weight=2.0, params={"std": 0.5})
    rw.tripod_pose_error = RewTerm(func=obs_mdp.tripod_pose_error, weight=-1.0)
    rw.tripod_contact = RewTerm(func=obs_mdp.tripod_contact, weight=1.0)
    rw.swing_clearance = RewTerm(func=obs_mdp.swing_clearance, weight=0.5, params={"target": 0.05})
    rw.foot_slip = RewTerm(func=obs_mdp.foot_slip, weight=-0.5)
    rw.track_ang_vel_z_exp.weight = 1.0
    rw.track_ang_vel_z_exp.params["std"] = 0.15


def use_real_masses(cfg: RoSpiderObstacleEnvCfg, scale_range: tuple[float, float] = (0.9, 1.1)) -> None:
    """무게 분포를 팀 실측(총 3.253 kg)에 맞춘다. 기존 '몸통에 +0.8~1.8 kg' 이벤트는 끈다.
    학습: 부위마다 ±10 % 무작위. 평가는 scale_range=(1, 1)."""
    cfg.events.add_base_mass = None
    cfg.events.real_masses = EventTerm(func=obs_mdp.set_real_masses, mode="startup",
                                       params={"scale_range": scale_range})
