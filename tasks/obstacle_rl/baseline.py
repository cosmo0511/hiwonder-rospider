"""기존 tripod 걸음(rospider_wasd.py) 을 Isaac Lab env 에서 돌리는 베이스라인 제어기. **Kit 이 뜬 뒤에 import.**"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv

TRIPOD_A = ("LF", "RM", "LR")

# ---- 기존 tripod 걸음. rospider_wasd.py 의 leg_targets() 를 그대로 옮김 (값을 바꾸지 말 것)
GAIT_HZ, COXA_AMP, FEMUR_LIFT, TIBIA_LIFT, SMOOTH_60HZ = 1.2, 0.35, 0.6, 0.3, 0.25
SIDE_SIGN = {"L": -1.0, "R": 1.0}


class TripodBaseline:
    """rospider_wasd.py 와 같은 열린 루프 tripod 걸음. 관절 목표를 절대각(rad)으로 만들어 행동으로 바꾼다."""

    def __init__(self, env: ManagerBasedRLEnv, random_phase: bool = False):
        if "joint_pos" not in env.action_manager.active_terms:
            raise SystemExit("[baseline] tripod 베이스라인은 관절 위치 행동(joint_pos) 실험에서만 돈다. --exp 2c_mass 처럼 주자")
        term = env.action_manager.get_term("joint_pos")
        self.names = list(term._joint_names)
        self.offset = term._offset if torch.is_tensor(term._offset) else torch.tensor(term._offset, device=env.device)
        self.scale = term._scale if torch.is_tensor(term._scale) else torch.tensor(term._scale, device=env.device)
        self.dt = env.step_dt
        # wasd 는 60 Hz 에서 프레임마다 SMOOTH 만큼 따라간다 -> 이 env 의 제어 주기로 환산
        self.alpha = 1.0 - (1.0 - SMOOTH_60HZ) ** (60.0 * self.dt)
        # 시작 박자를 env 마다 다르게 (모두 0 이면 반복 시도가 거의 같아져 통과율이 0 / 100 % 로 뭉친다)
        self.phase = torch.rand(env.num_envs, device=env.device) if random_phase else torch.zeros(env.num_envs, device=env.device)
        self.q = torch.zeros(env.num_envs, len(self.names), device=env.device)   # wasd 처럼 0 에서 시작

    def act(self, vx: float = 1.0, wz: float = 0.0) -> torch.Tensor:
        if abs(vx) + abs(wz) > 0:
            self.phase = (self.phase + GAIT_HZ * self.dt) % 1.0
        act = min(1.0, abs(vx) + abs(wz))
        des = torch.zeros_like(self.q)
        for j, name in enumerate(self.names):
            kind, leg = name.split("_")[0], name.split("_")[1]
            p = (self.phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
            swing = p < 0.5
            u = torch.where(swing, p / 0.5, (p - 0.5) / 0.5)
            frac = torch.where(swing, -torch.cos(math.pi * u), torch.cos(math.pi * u))
            lift = torch.where(swing, torch.sin(math.pi * u), torch.zeros_like(u))
            amp = max(-1.0, min(1.0, (vx - wz) if leg[0] == "L" else (vx + wz)))
            if kind == "coxa":
                des[:, j] = SIDE_SIGN[leg[0]] * COXA_AMP * amp * frac
            elif kind == "femur":
                des[:, j] = FEMUR_LIFT * act * lift
            else:
                des[:, j] = TIBIA_LIFT * act * lift
        self.q += self.alpha * (des - self.q)
        return (self.q - self.offset) / self.scale


