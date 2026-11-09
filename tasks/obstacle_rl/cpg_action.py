"""혼합 행동: 박자 신호기(CPG) 가 tripod 걸음을 만들고, 정책은 그 걸음을 조절 + 관절을 조금 보정한다.

참고: JetHexa(Hiwonder 6족) 를 CPG + RL 로 학습한 논문 (Biomimetics 2026, PMC13509619).
관절 18 개를 정책이 직접 조종하면 완주 20 %, CPG 를 정책이 조절하면 88 % 였다.

행동 21 개 = [걸음 크기, 걸음 빠르기, 좌우 회전, 관절 보정 18] (use_lift 면 22 개: 회전 뒤에 다리 드는 높이)
- 걸음 크기: 명령 속도로 정한 기본 크기 x (1 + 0.3 a0)
- 걸음 빠르기: 1.2 Hz x (1 + 0.3 a1)
- 좌우 회전: 왼쪽 coxa 스윙 x (1 + 0.25 a2), 오른쪽 x (1 - 0.25 a2)
- 관절 보정: 0.2 rad x a (장애물 앞에서 다리를 더 들거나 딛는 자리를 바꿀 여지)
- use_yaw 면 명령 회전 속도(wz)도 wasd A / D 키처럼 넣는다: 왼쪽 크기 - yaw, 오른쪽 + yaw (제자리 회전 가능).
  명령이 거의 0 이면 wasd 처럼 다리를 안 들고 선다 (act = 명령 크기 / stand_gate, 0 ~ 1)
관절 목표 = 저역통과(wasd tripod 궤적(박자, 크기, 회전)) + 보정. 행동이 0 이면 rospider_wasd.py 걸음과 같다
(wasd 처럼 0 중심 절대각, 60 Hz 에서 프레임당 0.25 따라가는 필터를 이 제어 주기로 환산). 박자는 env._gait_phase 에 둔다.
**Kit 이 뜬 뒤에 import.**
"""

from __future__ import annotations

import math
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import mdp as gait_mdp


class TripodCPGAction(ActionTerm):
    cfg: "TripodCPGActionCfg"

    def __init__(self, cfg: "TripodCPGActionCfg", env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        names = [f"{k}_{leg}_joint" for leg in gait_mdp.LEGS for k in ("coxa", "femur", "tibla")]
        self._joint_ids = self._asset.find_joints(names, preserve_order=True)[0]
        self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._targets = torch.zeros(self.num_envs, 18, device=self.device)
        self._q = torch.zeros(self.num_envs, 18, device=self.device)    # 저역통과된 걸음 목표 (wasd 의 q)
        self._dt = env.step_dt
        self._alpha = 1.0 - (1.0 - 0.25) ** (60.0 * self._dt)            # wasd SMOOTH 0.25 @ 60 Hz
        env._gait_phase = self._start_phase(self.num_envs)
        env._gait_act = torch.ones(self.num_envs, device=self.device)
        self._is_left = torch.tensor([leg[0] == "L" for leg in gait_mdp.LEGS], device=self.device)
        self._is_a = torch.tensor([leg in gait_mdp.TRIPOD_A for leg in gait_mdp.LEGS], device=self.device)
        self._side = torch.tensor([gait_mdp.SIDE_SIGN[leg[0]] for leg in gait_mdp.LEGS], device=self.device)

    @property
    def action_dim(self) -> int:
        return 21 + int(self.cfg.use_lift)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        env = self._env
        cmd = env.command_manager.get_command(self.cfg.command_name)
        base_amp = (cmd[:, 0] / gait_mdp.BASE_SPEED).clamp(0.0, 1.2)
        amp = base_amp * (1.0 + self.cfg.amp_range * a[:, 0])
        fmin, fmax = self.cfg.freq_scale
        hz = gait_mdp.GAIT_HZ * ((fmin + fmax) / 2 + (fmax - fmin) / 2 * a[:, 1])
        turn = self.cfg.turn_range * a[:, 2]
        k_lift = (1.0 + self.cfg.lift_range * a[:, 3]).unsqueeze(-1) if self.cfg.use_lift else 1.0
        if self.cfg.use_yaw:
            yaw = (cmd[:, 2] / self.cfg.base_yaw_rate).clamp(-1.0, 1.0)
            act = ((base_amp + yaw.abs()) / self.cfg.stand_gate).clamp(0.0, 1.0)
        else:
            yaw = torch.zeros_like(amp)
            act = torch.ones_like(amp)
        env._gait_act = act
        res = a[:, 3 + int(self.cfg.use_lift):]
        env._gait_phase = (env._gait_phase + hz * self._dt) % 1.0

        ph = env._gait_phase.unsqueeze(-1)                                  # (N, 1)
        p = (ph + torch.where(self._is_a, 0.0, 0.5)) % 1.0                  # (N, 6)
        swing = p < 0.5
        u = torch.where(swing, p / 0.5, (p - 0.5) / 0.5)
        frac = torch.where(swing, -torch.cos(math.pi * u), torch.cos(math.pi * u))
        lift = torch.where(swing, torch.sin(math.pi * u), torch.zeros_like(u))
        side_k = torch.where(self._is_left, 1.0 + turn.unsqueeze(-1), 1.0 - turn.unsqueeze(-1))
        # wasd: 왼쪽 크기 vx - wz, 오른쪽 vx + wz
        side_amp = (amp.unsqueeze(-1) * side_k + torch.where(self._is_left, -1.0, 1.0) * yaw.unsqueeze(-1)).clamp(-2.0, 2.0)   # 직진만일 때는 안 걸림 (최대 1.2 x 1.3 x 1.25)
        coxa = self._side * gait_mdp.COXA_AMP * side_amp * frac
        femur = gait_mdp.FEMUR_LIFT * lift * k_lift * act.unsqueeze(-1)
        tibla = gait_mdp.TIBIA_LIFT * lift * k_lift * act.unsqueeze(-1)
        gait = torch.stack([coxa, femur, tibla], dim=-1).reshape(self.num_envs, 18)
        self._q += self._alpha * (gait - self._q)
        self._targets[:] = self._q + self.cfg.residual_scale * res

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._q[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]
        n = self._env._gait_phase[env_ids].numel()
        self._env._gait_phase[env_ids] = self._start_phase(n)

    def _start_phase(self, n: int) -> torch.Tensor:
        """0 또는 0.5 에서 시작 (두 조 모두 발을 들기 직전이라 목표가 한 번에 뛰지 않음)."""
        return 0.5 * torch.randint(0, 2, (n,), device=self.device).float()


@configclass
class TripodCPGActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = TripodCPGAction
    asset_name: str = "robot"
    command_name: str = "base_velocity"
    amp_range: float = 0.3
    freq_scale: tuple[float, float] = (0.7, 1.3)
    """걸음 빠르기 배율 범위 (1.2 Hz 에 곱함). 행동 -1 -> 최소, +1 -> 최대."""
    turn_range: float = 0.25
    residual_scale: float = 0.2
    use_lift: bool = False
    """True 면 행동 하나를 더 받아 다리 드는 높이(femur / tibla 들기) 를 x (1 + lift_range a) 로 조절."""
    lift_range: float = 0.5
    use_yaw: bool = False
    """True 면 명령 wz 를 wasd A / D 처럼 좌우 반대 스윙으로 넣고, 명령이 0 근처면 선다 (3-k 정지 · 회전)."""
    base_yaw_rate: float = 0.5
    """wasd A 키(wz = 1) 가 평지에서 내는 회전 속도 [rad/s]. command_eval.py --controller tripod 로 잰다."""
    stand_gate: float = 0.3
    """(전진 크기 + |yaw|) 가 이 값보다 작으면 다리 드는 높이를 비례해서 줄임. 0.3 = 0.054 m/s. 직진 명령(0.14 이상)은 영향 없음."""
