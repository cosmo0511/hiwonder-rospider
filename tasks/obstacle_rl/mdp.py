"""장애물 과제 전용 MDP 함수 (이벤트, 보상, 종료). **Kit 이 뜬 뒤에 import 해야 한다.**

굴러가는 파이프
---------------
파이프는 지형이 아니라 env 마다 두 개씩 있는 RigidObject(pipe_0, pipe_1) 다. 리셋 때
그 env 가 놓인 타일 종류를 보고 자리를 정한다:
- 파이프가 있는 타일(pipes / map1 / map2): course.pipe_centers() 위치 + 작은 흔들림
- 나머지 타일: 로봇 뒤(course.PIPE_PARK)에 치워 둔다

타일 종류는 ``terrain.terrain_types``(env 마다 열 번호)와 지형 생성기 설정에서 계산한다.
커리큘럼 모드에서 열 번호가 env 마다 고정이라는 Isaac Lab 동작에 기댄다 [미검증].

주의: RayCaster(높이 스캔)는 정적 메시(/World/ground)만 본다. **움직이는 파이프는 높이 스캔에
안 잡힌다.** 정책은 발 접촉과 몸 기울기로만 파이프를 느낀다. 파이프를 보게 하려면
카메라 관측이나 파이프 위치(특권 관측)를 따로 넣어야 한다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from . import course
from .terrain_cfg import column_kinds

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_KIND_CACHE: dict[int, list[str]] = {}


def _env_kinds(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> list[str]:
    terrain = env.scene.terrain
    key = id(terrain.cfg.terrain_generator)
    if key not in _KIND_CACHE:
        _KIND_CACHE[key] = column_kinds(terrain.cfg.terrain_generator)
    cols = terrain.terrain_types[env_ids].tolist()
    return [_KIND_CACHE[key][int(c)] for c in cols]


def reset_pipes(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    jitter_x: float = 0.03,
    pipe_names: tuple[str, str] = ("pipe_0", "pipe_1"),
):
    """파이프 두 개를 그 env 의 타일 종류에 맞게 놓고 속도를 0 으로."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    kinds = _env_kinds(env, env_ids)
    origins = env.scene.env_origins[env_ids]
    n = len(env_ids)
    r = course.PIPE_D / 2
    for i, name in enumerate(pipe_names):
        obj: RigidObject = env.scene[name]
        dx = torch.empty(n, device=env.device)
        for e, k in enumerate(kinds):
            centers = course.pipe_centers(k)
            dx[e] = centers[i] if centers is not None else course.PIPE_PARK[i]
        has = torch.tensor([course.pipe_centers(k) is not None for k in kinds], device=env.device)
        dx = dx + torch.where(has, (torch.rand(n, device=env.device) * 2 - 1) * jitter_x, torch.zeros_like(dx))
        pos = origins.clone()
        pos[:, 0] += dx
        pos[:, 2] += r + 0.002
        quat = torch.zeros(n, 4, device=env.device)
        quat[:, 0] = 1.0   # 원통 축은 spawn 에서 y 로 잡아 둠
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def forward_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """명령 방향(몸 기준 전방)으로 실제 나아간 속도. 장애물 앞에서 멈춰 버티는 걸 줄이려고 둔다.

    속도 추종 보상(exp)은 장애물에 막혀 0 이 되어도 벌이 작아서, 넘는 대신 서 있는 정책이
    나오기 쉽다. 이 항은 명령 속도까지만 선형으로 준다(넘치는 속도는 보상 없음).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    vx = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(torch.minimum(vx, cmd), min=-0.1) * (cmd > 0.02)


def reached_platform(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """평가용: 로봇이 작업대(높이 0.24 - 0.03) 위로 올라섰는가. 계단이 없는 타일에서는 항상 False."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z > course.STAIR_N * course.STAIR_RISE - 0.03


# ---------------------------------------------------------------- tripod 걸음 학습용 (env_cfg.use_tripod_gait)
# 기준 걸음은 rospider_wasd.py 의 leg_targets() (실제로 걷는 걸 확인한 열린 루프 tripod).
# 박자 phase = (t * GAIT_HZ) % 1. 그룹 A(LF, RM, LR)는 phase < 0.5 에서 스윙, B 는 반 박자 뒤.

GAIT_HZ = 1.2
COXA_AMP, FEMUR_LIFT, TIBIA_LIFT = 0.35, 0.6, 0.3
BASE_SPEED = 0.18        # wasd 걸음(vx=1)이 평지에서 내는 속도 (gait_eval 측정 0.183 m/s)
LEGS = ("LF", "LM", "LR", "RF", "RM", "RR")
TRIPOD_A = ("LF", "RM", "LR")
SIDE_SIGN = {"L": -1.0, "R": 1.0}
# 발끝 = tibla_* 프레임 기준 end_*_joint 원점 (URDF). 발끝은 지름 18 mm 구.
FOOT_TIP = {
    "LF": (0.11519, -0.0097341, 0.001174),
    "LM": (0.11519, -0.0097478, -0.0020999),
    "LR": (0.11519, -0.0097478, -0.00039987),
    "RF": (0.11519, 0.009749, 0.0010888),
    "RM": (0.1107, 0.011377, 0.016938),
    "RR": (0.11519, 0.009749, 0.00093881),
}
_GAIT_CACHE: dict = {}


def _gait_ids(env: "ManagerBasedRLEnv") -> dict:
    """다리 순서(LEGS)대로 관절 / 바디 / 접촉 센서 인덱스. 처음 한 번만 찾는다."""
    if id(env) not in _GAIT_CACHE:
        robot: Articulation = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        joints = [f"{k}_{leg}_joint" for leg in LEGS for k in ("coxa", "femur", "tibla")]
        _GAIT_CACHE[id(env)] = {
            "joint": robot.find_joints(joints, preserve_order=True)[0],
            "body": [robot.find_bodies(f"tibla_{leg}")[0][0] for leg in LEGS],
            "sensor": [sensor.find_bodies(f"tibla_{leg}")[0][0] for leg in LEGS],
            "tip": torch.tensor([FOOT_TIP[leg] for leg in LEGS], device=env.device),
            "group_a": torch.tensor([leg in TRIPOD_A for leg in LEGS], device=env.device),
        }
    return _GAIT_CACHE[id(env)]


def _phase(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """걸음 박자 0 ~ 1. CPG 행동(cpg_action.py)을 쓰면 그 박자(빠르기가 바뀜), 아니면 1.2 Hz 고정 시계."""
    if hasattr(env, "_gait_phase"):
        return env._gait_phase
    return (env.episode_length_buf.float() * env.step_dt * GAIT_HZ) % 1.0


def _stance_wanted(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(env, 6) 이 박자에 땅에 있어야 하는 발."""
    a_swing = (_phase(env) < 0.5).unsqueeze(-1)
    return torch.where(_gait_ids(env)["group_a"], ~a_swing, a_swing)


def _feet(env: "ManagerBasedRLEnv"):
    """발끝 높이 z, 발끝 수평 속도 |v_xy|, 접촉 여부. 모두 (env, 6)."""
    ids = _gait_ids(env)
    robot: Articulation = env.scene["robot"]
    n = env.num_envs
    q = robot.data.body_quat_w[:, ids["body"]].reshape(-1, 4)
    off = math_utils.quat_apply(q, ids["tip"].repeat(n, 1)).reshape(n, 6, 3)
    pos = robot.data.body_pos_w[:, ids["body"]] + off
    vel = robot.data.body_lin_vel_w[:, ids["body"]] + torch.cross(robot.data.body_ang_vel_w[:, ids["body"]], off, dim=-1)
    contact = env.scene["contact_forces"].data.net_forces_w[:, ids["sensor"]].norm(dim=-1) > 1.0
    return pos[..., 2], vel[..., :2].norm(dim=-1), contact


def gait_clock(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """관측: 걸음 박자 sin, cos. 실기에서는 제어기 내부 시계로 같은 값을 만든다."""
    ph = 2 * math.pi * _phase(env)
    return torch.stack([torch.sin(ph), torch.cos(ph)], dim=-1)


def reference_joints(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """(env, 18) 기준 관절각 (rad). wasd leg_targets(phase, vx, 0) 에서 앞뒤 스윙 폭만 명령 속도에 비례."""
    vx = env.command_manager.get_command(command_name)[:, 0]
    amp = torch.clamp(vx / BASE_SPEED, 0.0, 1.0)
    ph = _phase(env)
    out = []
    for leg in LEGS:
        p = (ph + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
        swing = p < 0.5
        u = torch.where(swing, p / 0.5, (p - 0.5) / 0.5)
        frac = torch.where(swing, -torch.cos(math.pi * u), torch.cos(math.pi * u))
        lift = torch.where(swing, torch.sin(math.pi * u), torch.zeros_like(u))
        out += [SIDE_SIGN[leg[0]] * COXA_AMP * amp * frac, FEMUR_LIFT * lift, TIBIA_LIFT * lift]
    return torch.stack(out, dim=-1)


def imitate_tripod(env: "ManagerBasedRLEnv", std: float = 0.15, command_name: str = "base_velocity") -> torch.Tensor:
    """보상: 다리 관절이 기준 tripod 궤적을 따라가는 정도. exp(-평균 제곱오차 / std^2), 0 ~ 1."""
    q = env.scene["robot"].data.joint_pos[:, _gait_ids(env)["joint"]]
    err = (q - reference_joints(env, command_name)).pow(2).mean(dim=-1)
    return torch.exp(-err / std**2)


def tripod_contact(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """보상: 박자대로 닿고 / 떠 있는 발의 비율, 0 ~ 1."""
    _, _, contact = _feet(env)
    return (contact == _stance_wanted(env)).float().mean(dim=-1)


def swing_clearance(env: "ManagerBasedRLEnv", target: float = 0.05) -> torch.Tensor:
    """보상: 스윙 차례인 발끝이 디딘 발들보다 얼마나 높이 떴나 (target 에서 1 로 포화), 0 ~ 1.
    기준을 땅이 아니라 가장 낮은 발끝으로 잡아서 장애물 지형에서도 그대로 쓴다."""
    z, _, _ = _feet(env)
    lift = (z - z.min(dim=-1, keepdim=True).values).clamp(0.0, target) / target
    swing = ~_stance_wanted(env)
    return (lift * swing).sum(dim=-1) / swing.sum(dim=-1).clamp(min=1)


def foot_slip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """벌점용: 땅에 닿은 발끝의 수평 속도 합 (m/s)."""
    _, v, contact = _feet(env)
    return (v * contact).sum(dim=-1)


def tripod_pose_error(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """벌점용: 기준 tripod 궤적과의 관절 평균 제곱오차 (rad^2). imitate_tripod(exp)는 멀리 떨어지면 0 에 붙어
    방향을 못 알려 주므로, 멀어도 기울기가 남는 제곱 오차를 같이 쓴다."""
    q = env.scene["robot"].data.joint_pos[:, _gait_ids(env)["joint"]]
    return (q - reference_joints(env, command_name)).pow(2).mean(dim=-1)


# ---------------------------------------------------------------- 실측 무게 분포 (env_cfg.use_real_masses)
# 2026-11-07 팀 실측 (g): 몸통 1418, 배터리 254, 충전 단자 22, 부품 8, 다리 1개 201
#   (coxa 14 + femur 27 + tibla+발끝 38 + 서보 59 x 2), 팔 전체 402, 저울 합계 3253.
# 다리 서보 2개는 femur, tibla 마디에 하나씩 (URDF 메시로 확인: coxa 메시는 브래킷뿐, femur / tibla 메시에 서보 몸통).
# coxa 서보는 몸통 쪽 leg_center 에 있어 몸통 측정에 포함. 부위 합 3310 g 을 저울 3253 g 에 맞춰 0.983 배.
REAL_MASS_G = {
    "body": 1673.0,
    "coxa": 14.0,     # 다리 하나당
    "femur": 86.3,
    "tibla": 97.3,
    "arm": 395.0,
}
ARM_BODY_RE = r"(link[1-5]|gripper_link|[rl]_.*link|camera_connect_link|depth_cam_link)"


def _mass_groups(robot: Articulation) -> dict[str, list[int]]:
    import re  # noqa: PLC0415

    groups: dict[str, list[int]] = {k: [] for k in REAL_MASS_G}
    for i, name in enumerate(robot.body_names):
        if name.startswith(("coxa_", "femur_", "tibla_")):
            groups[name.split("_")[0]].append(i)
        elif re.fullmatch(ARM_BODY_RE, name):
            groups["arm"].append(i)
        else:
            groups["body"].append(i)
    return groups


def set_real_masses(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] = (1.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """시작할 때 한 번: 부위(몸통, coxa, femur, tibla, 팔)별 총질량을 실측값에 맞춘다.
    부위 안에서는 URDF 비율을 유지하고, 관성도 같은 배율로 늘린다. scale_range 로 부위마다 env 별 무작위 배율."""
    robot: Articulation = env.scene[asset_cfg.name]
    view = robot.root_physx_view
    masses = view.get_masses().clone()          # (env, body), CPU
    inertias = view.get_inertias().clone()      # (env, body, 9)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs)
    env_ids = env_ids.cpu()
    groups = _mass_groups(robot)
    for g, idx in groups.items():
        if not idx:
            continue
        per_leg = g in ("coxa", "femur", "tibla")
        target = REAL_MASS_G[g] / 1000.0 * (len(idx) if per_leg else 1)
        cur = masses[env_ids][:, idx].sum(dim=-1, keepdim=True)
        k = target / cur * torch.empty_like(cur).uniform_(*scale_range)
        masses[env_ids[:, None], torch.tensor(idx)[None, :]] = masses[env_ids][:, idx] * k
        inertias[env_ids[:, None], torch.tensor(idx)[None, :]] = inertias[env_ids][:, idx] * k.unsqueeze(-1)
    view.set_masses(masses, env_ids)
    view.set_inertias(inertias, env_ids)
    if not getattr(env, "_real_mass_logged", False):
        env._real_mass_logged = True
        m0 = masses[env_ids[0]]
        print("[real_mass] " + ", ".join(f"{g} {float(m0[idx].sum()) * 1000:.0f} g ({len(idx)} 바디)" for g, idx in groups.items()),
              f"| 합계 {float(m0.sum()) * 1000:.0f} g", flush=True)
        print(f"[real_mass] 팔 바디: {[robot.body_names[i] for i in groups['arm']]}", flush=True)


# ---------------------------------------------------------------- 2-d 후보 (검토 AI 지적 반영)
def track_vx_exp(env: "ManagerBasedRLEnv", std: float = 0.06, command_name: str = "base_velocity") -> torch.Tensor:
    """보상: 전진 속도(vx)만 따로 추종. 기본 track_lin_vel_xy_exp 는 vx 오차가 크면 vy 기울기까지 0 이 된다."""
    asset: Articulation = env.scene["robot"]
    err = env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-err.square() / std**2)


def lin_vel_y_l2(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """벌점용: 몸 기준 옆 속도와 명령 옆 속도 차이의 제곱."""
    asset: Articulation = env.scene["robot"]
    return (env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1]).square()


def stance_slip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """벌점용: 디딜 차례인 발 중 닿아 있는 발의 미끄러짐 합 (m/s). foot_slip 과 달리 발을 띄워서 피할 수 없게
    stance_air 와 같이 쓴다."""
    _, v, contact = _feet(env)
    return (v * contact * _stance_wanted(env)).sum(dim=-1)


def stance_air(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """벌점용: 디딜 차례인데 떠 있는 발의 비율 (0 ~ 1). 두 번 톡톡 딛기, 튀어 다니기 방지."""
    _, _, contact = _feet(env)
    want = _stance_wanted(env)
    return ((~contact) & want).float().sum(dim=-1) / want.float().sum(dim=-1).clamp(min=1)


def imitate_tripod_stand(env: "ManagerBasedRLEnv", std: float = 0.5, command_name: str = "base_velocity") -> torch.Tensor:
    """imitate_tripod 와 같지만 기준 = 선 자세 + wasd 움직임. RM 다리처럼 선 자세가 0 이 아닌 다리를 비대칭으로 끌지 않는다."""
    robot: Articulation = env.scene["robot"]
    ids = _gait_ids(env)["joint"]
    ref = robot.data.default_joint_pos[:, ids] + reference_joints(env, command_name)
    err = (robot.data.joint_pos[:, ids] - ref).pow(2).mean(dim=-1)
    return torch.exp(-err / std**2)


def tripod_pose_error_stand(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ids = _gait_ids(env)["joint"]
    ref = robot.data.default_joint_pos[:, ids] + reference_joints(env, command_name)
    return (robot.data.joint_pos[:, ids] - ref).pow(2).mean(dim=-1)


def track_vy_exp(env: "ManagerBasedRLEnv", std: float = 0.04, command_name: str = "base_velocity") -> torch.Tensor:
    """보상: 옆 속도(vy) 추종을 vx 와 떼어서. 옆으로 새는 걸 vx 오차와 상관없이 잡는다."""
    asset: Articulation = env.scene["robot"]
    err = env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1]
    return torch.exp(-err.square() / std**2)


def heading_error(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """관측: 목표 방향 - 몸 방향 (rad, -pi ~ pi). 실기에서는 IMU 요 각속도를 적분해 얻는다 (JetHexa 논문과 같음)."""
    term = env.command_manager.get_term(command_name)
    err = math_utils.wrap_to_pi(term.heading_target - env.scene["robot"].data.heading_w)
    return err.unsqueeze(-1)


def heading_error_l2(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """벌점용: 방향 오차 제곱."""
    return heading_error(env, command_name).squeeze(-1).square()


def park_pipes(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, pipe_names: tuple[str, str] = ("pipe_0", "pipe_1")):
    """파이프를 로봇 뒤 주차 자리(course.PIPE_PARK)에 둔다. 파이프가 없는 지형(2단계, 평가)에서 reset_pipes 대신."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    origins = env.scene.env_origins[env_ids]
    n = len(env_ids)
    for i, name in enumerate(pipe_names):
        obj: RigidObject = env.scene[name]
        pos = origins.clone()
        pos[:, 0] += course.PIPE_PARK[i]
        pos[:, 2] += course.PIPE_D / 2 + 0.002
        quat = torch.zeros(n, 4, device=env.device)
        quat[:, 0] = 1.0
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
