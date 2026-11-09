"""실험 설정 모음. 이름 하나로 환경 + 에이전트 설정을 정한다 (train.py / gait_eval.py 의 --exp).

새 실험은 앞 실험을 불러 놓고 바뀐 것만 덮어쓴다. 무엇을 왜 바꿨는지는 REWARD_DESIGN.md 실험 기록에 적는다.
**Kit 이 뜬 뒤에 import.**
"""

from __future__ import annotations

import copy

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from . import mdp
from .cpg_action import TripodCPGActionCfg
from .robot_cfg import LEG_JOINTS
from .env_cfg import make_blind, make_flat, use_real_masses, use_tripod_gait, use_wasd_gains


def _exp1(env, agent):
    """실험 1: 평지, 이동 점수만."""
    make_flat(env)
    make_blind(env)
    use_wasd_gains(env)
    agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}


def _exp2b(env, agent):
    """실험 2-b: 박자 + 기존 tripod 걸음 따라 하기 (허용 0.5 rad, 관절 차이 벌점)."""
    make_flat(env)
    use_tripod_gait(env)   # make_blind 보다 먼저 (critic 에도 박자)
    make_blind(env)
    use_wasd_gains(env)
    agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}


def _exp2c(env, agent):
    """실험 2-c: 2-b 에서 걸음 따라 하기는 길잡이로만 줄이고, 속도 / 방향 맞추기를 키운다.
    2-b 결과: 걸음은 생겼지만 0.18 명령에 0.33 m/s, 옆으로 1.7 m 샘."""
    _exp2b(env, agent)
    rw = env.rewards
    rw.imitate_tripod.weight = 0.5
    rw.tripod_pose_error.weight = -0.25
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_lin_vel_xy_exp.params["std"] = 0.1
    rw.track_ang_vel_z_exp.weight = 2.0


def _exp2c_mass(env, agent):
    """2-c + 실측 무게 분포 (몸통 1673 g, 다리 197 g x 6, 팔 395 g). 무게만 바꿔서 2-c 와 비교."""
    _exp2c(env, agent)
    use_real_masses(env)


def _exp2d(env, agent):
    """2-d: 2c_mass + 검토 AI 지적 반영.
    (a) 속도 보상 정리: 앞으로 나아가기(비대칭, 과속에 벌 없음) 끔, vx / vy 추종을 따로
    (b) 미끄러짐 벌점은 디딜 차례인 발만 + 디딜 차례에 뜬 발 벌점 (발을 띄워 벌점 피하기 막음)
    (c) [취소] 따라 하기 기준을 선 자세 + wasd 로 바꾸면 검증된 걸음이 아니게 돼서 (검토 AI 2차) 원래 기준 유지
    (e) 위아래 출렁임 벌점 -0.5 -> -2.0 (튀는 걸음 방지)"""
    _exp2c_mass(env, agent)
    rw = env.rewards
    rw.forward_progress = None
    rw.track_lin_vel_xy_exp = None
    rw.track_vx = RewTerm(func=mdp.track_vx_exp, weight=2.0, params={"std": 0.06})
    rw.track_vy = RewTerm(func=mdp.track_vy_exp, weight=1.0, params={"std": 0.04})
    rw.foot_slip = None
    rw.stance_slip = RewTerm(func=mdp.stance_slip, weight=-0.5)
    rw.stance_air = RewTerm(func=mdp.stance_air, weight=-1.0)
    rw.tripod_contact.weight = 0.5      # stance_air 와 겹쳐서 낮춤 (검토 AI 2차)
    rw.lin_vel_z_l2.weight = -2.0


def _exp2e(env, agent):
    """2-e: 2-d + 정책이 최근 4 스텝 관측을 같이 봄 (블라인드라 속도 / 미끄러짐을 히스토리로 짐작)."""
    _exp2d(env, agent)
    env.observations.policy.history_length = 4


def _exp2f(env, agent):
    """2-f: 혼합 방식 (JetHexa 논문). CPG 가 tripod 걸음을 만들고 정책은 걸음 크기 · 빠르기 · 회전 + 관절 보정 0.2 rad.
    걸음 모양은 CPG 가 보장하므로 따라 하기 보상은 끔. 관측에 방향 오차 추가, 방향 오차 벌점 -2.0.
    속도 보상 / 디딤 벌점 / 무게는 2-d 와 같음."""
    make_flat(env)
    use_tripod_gait(env)
    env.observations.policy.heading_error = ObsTerm(func=mdp.heading_error)
    make_blind(env)
    use_wasd_gains(env)
    use_real_masses(env)
    agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    env.actions.joint_pos = None
    env.actions.cpg = TripodCPGActionCfg()
    rw = env.rewards
    rw.imitate_tripod = None
    rw.tripod_pose_error = None
    rw.forward_progress = None
    rw.track_lin_vel_xy_exp = None
    rw.track_vx = RewTerm(func=mdp.track_vx_exp, weight=2.0, params={"std": 0.06})
    rw.track_vy = RewTerm(func=mdp.track_vy_exp, weight=1.0, params={"std": 0.04})
    rw.track_ang_vel_z_exp.weight = 2.0
    rw.heading = RewTerm(func=mdp.heading_error_l2, weight=-2.0)
    rw.foot_slip = None
    rw.stance_slip = RewTerm(func=mdp.stance_slip, weight=-0.5)
    rw.stance_air = RewTerm(func=mdp.stance_air, weight=-1.0)
    rw.tripod_contact.weight = 0.5
    rw.lin_vel_z_l2.weight = -2.0


def _exp2g(env, agent):
    """2-g: 2-f + 장애물 단계에 필요한 행동 / 관측을 미리 넣고 평지에서 다시 학습 (차원이 바뀌면 이어 학습이 안 되므로 한 번에).
    - 행동 22: 다리 드는 높이 x (1 +- 0.5) 추가 (방지턱 8 cm, 계단 6 cm + 앞 턱. 평지 들림 7.3 cm)
    - 걸음 빠르기 범위 0.7 ~ 1.3 -> 0.5 ~ 1.3 배 (계단 / 파이프에서 천천히)
    - 정책 관측에 최근 4 스텝 기억 (블라인드가 발 걸림 / 단차를 알아챌 단서)"""
    _exp2f(env, agent)
    env.actions.cpg.use_lift = True
    env.actions.cpg.freq_scale = (0.5, 1.3)
    env.observations.policy.history_length = 4


def _stage2(env):
    """2단계 공통: 방지턱 / 구덩이 지형 + 커리큘럼. 검토 AI 3차 지적 반영."""
    from .stage2_terrain import STAGE2_TERRAINS_CFG  # noqa: PLC0415

    env.scene.terrain.terrain_type = "generator"
    env.scene.terrain.terrain_generator = copy.deepcopy(STAGE2_TERRAINS_CFG)
    env.scene.terrain.max_init_terrain_level = 2
    env.curriculum.terrain_levels = CurrTerm(func=loco_mdp.terrain_levels_vel)
    env.events.reset_pipes = EventTerm(func=mdp.park_pipes, mode="reset")   # 파이프는 로봇 뒤에 치움
    env.episode_length_s = 20.0          # 20 s x 0.14 ~ 0.20 m/s = 2.8 ~ 4.0 m. 2 m 넘게 가면 한 단계 위로
    env.commands.base_velocity.ranges.lin_vel_x = (0.14, 0.20)
    env.rewards.lin_vel_z_l2.weight = -0.5         # 방지턱을 오르내리는 위아래 움직임은 필요
    env.rewards.flat_orientation_l2.weight = -0.1  # 경사에서 숙이는 건 조금 허용
    env.events.physics_material.params["static_friction_range"] = (0.4, 1.0)   # 베이스라인이 마찰 0.5 에서 약함
    env.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.9)


def _exp3a(env, agent):
    """3-a (2단계): 2-g 정책을 이어 받아 방지턱 / 구덩이 지형에서 학습. 모양 · 폭을 섞고 높이는 낮게 시작해
    방지턱 10 cm, 구덩이 7 cm 까지 (목표 8 / 5 cm 보다 여유). 2 m 넘게 가면 다음 행(더 높은 장애물).
    실행: run_exp.sh 3a <iters> 0.18 <2-g 체크포인트>"""
    _exp2g(env, agent)
    _stage2(env)


def _exp3b(env, agent):
    """3-b (3단계 블라인드 시도): 3-a 정책을 이어 받아 팀 코스 지형 전체(방지턱, 구덩이, 굴러가는 파이프, 그레이팅 계단 +
    작업대, 맵 1 / 2) 에서 학습. 높이는 0.5 배부터 실물까지 커리큘럼. 블라인드로 안 되면 teacher-student 로 간다.
    실행: run_exp.sh 3b <iters> 0.18 <3-a 체크포인트>"""
    from isaaclab.managers import EventTermCfg as EventTerm  # noqa: PLC0415

    from .terrain_cfg import OBSTACLE_TERRAINS_CFG, enable_height_curriculum  # noqa: PLC0415

    _exp2g(env, agent)
    _stage2(env)
    from .stage2_terrain import Stage2TileCfg  # noqa: PLC0415

    gen = copy.deepcopy(OBSTACLE_TERRAINS_CFG)
    enable_height_curriculum(gen, low=0.5)
    # 3-a 에서 배운 원호 방지턱 / 상자 구덩이(폭 무작위, 10 / 7 cm)를 잊지 않게 조금 남긴다 (검토 AI 5차)
    for k, p in {"flat": 0.04, "bump": 0.12, "pit": 0.12, "stairs": 0.16, "map1": 0.15, "map2": 0.15}.items():
        gen.sub_terrains[k].proportion = p
    gen.sub_terrains["arc_bump"] = Stage2TileCfg(kind="arc_bump", proportion=0.08)
    gen.sub_terrains["box_pit"] = Stage2TileCfg(kind="box_pit", proportion=0.08)
    gen.num_cols = 18
    env.scene.terrain.terrain_generator = gen
    env.scene.terrain.max_init_terrain_level = 1
    env.events.reset_pipes = EventTerm(func=mdp.reset_pipes, mode="reset", params={"jitter_x": 0.03})
    env.episode_length_s = 30.0          # 계단 / 작업대까지 2.4 ~ 2.8 m, 계단에서 느려짐
    # 기본 커리큘럼(2 m 이상 걸었나)은 맵에서 작업대 도착(1.85 m)을 실패로 본다 -> 목표 지점 기준으로
    env.curriculum.terrain_levels = CurrTerm(func=mdp.course_curriculum)
    # 목표 도착 = 진짜 끝(성공). time_out=True 로 두면 rsl_rl 이 그 스텝 점수에 "지금 상태의 가치 x 0.99" 를 더해서
    # 도착이 잦아지면(77 %) critic 이 자기 값을 키우는 되먹임이 생긴다 -> 3-c 4303 / 4407 회 발산.
    # 대신 도착 보너스 2.0 (= 계속 걸었으면 받았을 점수 정도: 스텝당 약 0.015 / (1 - 0.99)).
    env.terminations.reached_goal = DoneTerm(func=mdp.reached_goal, time_out=False)
    env.rewards.goal_bonus = RewTerm(func=mdp.reached_goal, weight=2.0 / (env.sim.dt * env.decimation))
    env.terminations.physics_blowup = DoneTerm(func=mdp.physics_blowup)
    # CPG 행동은 안에서 ±1 로 잘라 쓰는데, 자르기 전 값(100 ~ 180 까지 튐)이 "직전 행동" 관측으로 들어가 critic 을 흔들었다.
    agent.clip_actions = 1.0


def _add_perception(env, agent, noise_scale: float = 1.0):
    """정책에 몸 앞 높이 지도 (x -0.15 ~ 0.75 m, y ±0.25 m, 5 cm 간격 = 19 x 11 = 209 칸) 를 보여준다.
    실물에서는 손목 뎁스 카메라(12 fps) + 다리 odometry 로 같은 격자를 만든다는 가정. 잡음 / 지연 / 칸 누락 포함.
    critic 에는 깨끗한 지도 + 파이프 위치 · 속도 (특권)."""
    from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: PLC0415
    from isaaclab.sensors import RayCasterCfg, patterns  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    env.scene.actor_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        offset=RayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.90, 0.50)),
        mesh_prim_paths=["/World/ground"],
        update_period=1.0 / 12.0,
        drift_range=(-0.03, 0.03),
    )

    @configclass
    class PerceptionCfg(ObsGroup):
        scan = ObsTerm(func=mdp.PerceptiveScan, params={"sensor_name": "actor_scanner", "noise": True,
                                                        "noise_scale": noise_scale})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivCfg(ObsGroup):
        scan = ObsTerm(func=mdp.PerceptiveScan, params={"sensor_name": "actor_scanner", "noise": False})
        pipes = ObsTerm(func=mdp.pipe_states)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    env.observations.perception = PerceptionCfg()
    env.observations.priv = PrivCfg()
    agent.obs_groups = {"policy": ["policy", "perception"], "critic": ["critic", "priv"]}


def _exp3c(env, agent):
    """3-c (3단계, 지형을 보는 정책): 3-b 와 같은 코스 지형 + 정책이 몸 앞 높이 지도(잡음 포함)를 봄.
    3-b 가중치에 새 입력 칸을 0 으로 붙여 이어 받는다 (expand_ckpt.py). 실행: run_exp.sh 3c <iters> 0.18 <확장한 체크포인트>"""
    _exp3b(env, agent)
    _add_perception(env, agent)



def _exp3d(env, agent):
    """3-d: 3-c 를 이어 받아 두 가지만 고침 (검토 AI 6차: 원인을 가르려고 속도 보너스는 아직 안 넣음).
    - 물리 폭발의 관절 속도 기준 60 -> 500 rad/s. 정상 걸음에서도 정강이가 착지 때 60 ~ 146 rad/s 로 튀어
      (기존걸음도 평균 80) 학습 종료의 7.5 % 가 가짜 실패였다. 이걸 피하려고 천천히 / 발 높이 걷게 됐을 수 있음
    - 평지 비율 0.04 -> 0.08 (맵 1 / 2 에서 0.02 씩)
    실행: run_exp.sh 3d <iters> 0.18 <3-c 체크포인트>"""
    _exp3c(env, agent)
    env.terminations.physics_blowup.params = {"max_joint_vel": 500.0}
    subs = env.scene.terrain.terrain_generator.sub_terrains
    subs["flat"].proportion = 0.08
    subs["map1"].proportion = 0.13
    subs["map2"].proportion = 0.13



def _exp3e(env, agent):
    """3-e: 3-d 를 이어 받아 평지 속도를 되살린다. 3-d 에서 물리 폭발 기준을 고쳐도 평지 0.080 m/s 그대로였음.
    원인: 전진 속도 보상 std 0.06 이 좁아서 0.08 m/s (오차 0.1) 에서는 보상이 거의 0 -> 빨라질 쪽으로 기울기가 없다.
    - 넓은 전진 속도 보상 추가 (std 0.15, 가중치 1.0): 오차 0.1 에서 0.64, 느린 걸음에서도 빨라지는 방향을 알려줌
    - 도착 보너스를 평균 속도 비율로 (0.5 + 0.5 x 비율)
    실행: run_exp.sh 3e <iters> 0.18 <3-d 체크포인트>"""
    _exp3d(env, agent)
    env.rewards.track_vx_coarse = RewTerm(func=mdp.track_vx_exp, weight=1.0, params={"std": 0.15})
    env.rewards.goal_bonus = RewTerm(func=mdp.goal_bonus_speed, weight=2.0 / (env.sim.dt * env.decimation))



def _exp3f(env, agent):
    """3-f: 3-e 와 같은 설정. 다른 점은 train.py 에서 행동 잡음 std 를 0.5 로 되돌리고 매번 0.8 이하로 묶는 것.
    3-c 에서 행동을 ±1 로 자르기 시작한 뒤 std 가 1 -> 4 (3-c 끝) -> 10 ~ 15 (3-e) 로 커져서,
    학습 중 행동이 거의 ±1 무작위였다 -> 평균 행동(평가)의 평지 속도가 보상을 바꿔도 0.08 m/s 에서 안 움직였다.
    실행: TRAIN_EXTRA="--std_reset 0.5 --std_max 0.8" run_exp.sh 3f <iters> 0.18 <3-e 체크포인트>"""
    _exp3e(env, agent)



def _exp3g(env, agent):
    """3-g: 땅 못 보는(블라인드) 정책을 계단까지 학습. 실물에는 높이 지도가 없으니 카메라 없이 어디까지 되는지 본다.
    3-a 는 방지턱 / 구덩이만 배웠고 계단은 한 번도 안 배웠다 (3-b 를 건너뜀). 3-b 코스 지형 + 3-d / 3-e / 3-f 에서 고친 것:
    물리 폭발 관절 기준 500, 평지 0.08, 넓은 속도 보상, 속도 비례 도착 보너스, std 상한 (train.py --std_max).
    실행: TRAIN_EXTRA="--std_max 0.8" run_exp.sh 3g <iters> 0.18 <3-a 체크포인트>"""
    _exp3b(env, agent)
    env.terminations.physics_blowup.params = {"max_joint_vel": 500.0}
    subs = env.scene.terrain.terrain_generator.sub_terrains
    subs["flat"].proportion = 0.08
    subs["map1"].proportion = 0.13
    subs["map2"].proportion = 0.13
    env.rewards.track_vx_coarse = RewTerm(func=mdp.track_vx_exp, weight=1.0, params={"std": 0.15})
    env.rewards.goal_bonus = RewTerm(func=mdp.goal_bonus_speed, weight=2.0 / (env.sim.dt * env.decimation))


def _exp3h(env, agent):
    """3-h: 3-g 그대로 + 정책 평균이 ±1 밖으로 나간 만큼 끌어당김 (train.py --bound_coef).
    3-c / 3-f / 3-g 모두 주기 / 회전 / 보폭 출력이 ±1 밖에 붙어 있었다 (3-f 평지 주기 평균 -38).
    샘플이 전부 같은 값으로 잘려 그 차원은 기울기가 0 -> 평지에서도 최저 주기로 느리게 걷는 원인 후보.
    실행: TRAIN_EXTRA="--std_max 0.8 --bound_coef 0.1" run_exp.sh 3h <iters> 0.18 <3-g 체크포인트>"""
    _exp3g(env, agent)



def _exp3i(env, agent):
    """3-i: 3-h 에서 잡음 줄임. 평균을 ±1 안으로 끌어오니 std 0.75 잡음이 실제로 걸음을 흔들어 (회전 · 옆 속도 보상 급락)
    학습 중 보상이 3-g 의 절반이 됨. 검토 AI 제안대로 std 0.4 로 되돌리고 상한 0.5, 엔트로피 0.005 -> 0.002.
    실행: TRAIN_EXTRA="--std_reset 0.4 --std_max 0.5 --bound_coef 0.1" run_exp.sh 3i <iters> 0.18 <3-h model_6600>"""
    _exp3h(env, agent)
    agent.algorithm.entropy_coef = 0.002



def _exp3j(env, agent):
    """3-j: 3-i 에 실물보다 높은 계단까지. 3-i 는 단 6 cm 에서만 배워서 7 cm 85 %, 8 cm 40 %, 9 cm 5 % 로 여유가 작았다.
    코스 타일 높이 배율을 0.5 ~ 1.0 -> 0.5 ~ 1.5 (단 3 ~ 9 cm, 방지턱 · 구덩이도 같이 커짐). 실물(1.0)은 중간 난이도가 된다.
    2단계 타일(arc_bump / box_pit)은 그대로.
    실행: TRAIN_EXTRA="--std_max 0.5 --bound_coef 0.1" run_exp.sh 3j <iters> 0.18 <3-i 체크포인트>"""
    from .terrain_cfg import RoSpiderTileCfg  # noqa: PLC0415

    _exp3i(env, agent)
    for sub in env.scene.terrain.terrain_generator.sub_terrains.values():
        if isinstance(sub, RoSpiderTileCfg):
            sub.height_scale_range = (0.5, 1.5)


def _exp3k(env, agent):
    """3-k: 3-j 에 정지 · 회전 명령. 지금까지는 직진(0.14 ~ 0.20 m/s)만 배워서 도착 뒤에도 멈추지 않고, 실물에서 조종할 수 없다.
    - CPG: 명령 wz 를 wasd A / D 키처럼 좌우 반대 스윙으로 (제자리 회전), 명령이 0 근처면 다리를 안 들고 선다 (cpg_action use_yaw)
    - 명령 (mdp.MixedVelocityCommand): 평지 타일은 직진 25 % / 걸으며 돌기 20 % / 제자리 돌기 30 % / 정지 25 %,
      코스 타일은 10 % 정지 (계단 · 파이프 위에서도 서기). 4 ~ 8 초마다 다시 뽑음
    - 평지 타일 비율 0.08 -> 0.30 (명령 연습 자리)
    - 정지 중에는 여섯 발 모두 디딜 차례로 (stance_air / stance_slip / tripod_contact 가 서 있기를 보상)
    관측 · 행동 크기는 3-j 와 같아서 그대로 이어 받는다.
    실행: TRAIN_EXTRA="--std_max 0.5 --bound_coef 0.1" run_exp.sh 3k <iters> 0.18 <3-j 체크포인트>"""
    _exp3j(env, agent)
    env.actions.cpg.use_yaw = True
    env.actions.cpg.base_yaw_rate = BASE_YAW_RATE
    cmd = mdp.use_mixed_commands(env.commands.base_velocity)
    cmd.resampling_time_range = (4.0, 8.0)
    cmd.ranges.ang_vel_z = (-0.5, 0.5)
    env.commands.base_velocity = cmd
    env.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.30


def _exp3l(env, agent):
    """3-l: 3-k 에 다리 관절 armature (서보 기어 관성). 지금까지 armature 0 이라 착지 순간 정강이가 한 물리 스텝 동안
    60 ~ 90 rad/s 로 튐 (PhysX 속도 제한 6 rad/s 를 넘김. 실물 HX-35H 는 0.18 s/60° = 약 5.8 rad/s).
    평가에서 armature 0.01 이면 튐이 사라지고 (최대 6.0 rad/s) 발 미끄러짐도 17 -> 8 cm/s 로 줄었다.
    대신 armature 없이 배운 3-k 는 옆으로 더 샘 (2 -> 6 cm) -> armature 를 넣고 이어서 학습.
    실물 값은 모르니 0.01 기준 0.5 ~ 2배 (0.005 ~ 0.02) 로 로봇마다 다르게."""
    _exp3k(env, agent)
    env.scene.robot.actuators["legs"].armature = 0.01
    env.events.leg_armature = EventTerm(
        func=loco_mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
                "armature_distribution_params": (0.5, 2.0), "operation": "scale", "distribution": "log_uniform"})


BASE_YAW_RATE = 0.81   # wasd A 키(wz = 1) 제자리 회전 속도 rad/s. command_eval.py --controller tripod 로 잼 (8대 평균 0.812)


EXPERIMENTS = {
    "1": _exp1,
    "2b": _exp2b,
    "2c": _exp2c,
    "2c_mass": _exp2c_mass,
    "2d": _exp2d,
    "2e": _exp2e,
    "2f": _exp2f,
    "2g": _exp2g,
    "3a": _exp3a,
    "3b": _exp3b,
    "3c": _exp3c,
    "3d": _exp3d,
    "3e": _exp3e,
    "3f": _exp3f,
    "3g": _exp3g,
    "3h": _exp3h,
    "3i": _exp3i,
    "3j": _exp3j,
    "3k": _exp3k,
    "3l": _exp3l,
}


def apply(name: str, env_cfg, agent_cfg) -> None:
    if name not in EXPERIMENTS:
        raise SystemExit(f"모르는 실험 '{name}'. 있는 것: {', '.join(EXPERIMENTS)}")
    EXPERIMENTS[name](env_cfg, agent_cfg)
