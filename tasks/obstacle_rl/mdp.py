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

from typing import TYPE_CHECKING

import torch

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
