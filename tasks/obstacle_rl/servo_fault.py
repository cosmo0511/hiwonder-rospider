"""18 Hz 제어 + 실물 버스 서보(HX-35H) 흉내 + 다리 고장. **Kit 이 뜬 뒤에 import.**

experiments.py 의 r18 / f18 이 쓴다. 자세한 배경은 FAULT_18HZ.md.

왜 필요한가
-----------
- HX-35H 는 115200 baud 반이중 버스에서 서보 하나씩 묻고 답한다 (여러 개 한 번에 읽는 명령 없음).
  다리 18 개를 매번 읽고 쓰면 선 위 시간만 약 38 ms, 응답 지연까지 더하면 약 55 ms -> 약 18 Hz.
  그래서 정책을 18 Hz (물리 0.005 s x 11 = 0.055 s) 로 돌린다.
- 서보는 위치만 준다 (속도 · 전류 없음). 정책의 관절 속도 관측은 실물처럼 **읽은 각도의 차분** 으로 만든다.
- 서보는 MOVE_TIME 동안 목표까지 직선으로 간다 -> 시뮬도 목표를 제어 주기 동안 직선 보간.
- 읽기 지연 (18 개를 차례로 읽어서 관절마다 0 ~ 1 주기 늦음), 0점 오차 (위키 8.1 서보 편차 보정),
  포텐셔미터 눈금 0.24 도 (0 ~ 1000 -> 0 ~ 240 도).

다리 고장 (f18)
---------------
에피소드마다 확률 p_fault 로 다리 하나 (LF ~ RR 무작위) 가 고장. 다리 전체 또는 관절 하나.
- limp  : 힘 빠짐. 강성 0, 기어 마찰 정도의 감쇠만 (서보 전원 / 통신 끊김)
- locked: 굳음. 무작위 각도에 고정 (기어 걸림)
- weak  : 토크 15 ~ 50 % (과열 · 열화)
힘 빠진(limp) 관절의 읽기값은 0 으로 가린다 (죽은 서보는 위치도 못 준다고 가정). 어느 다리가 고장인지는
'fault' 관측 그룹 (one-hot 6) 으로 준다 — 실물에서 왼쪽 앞이 고장인 건 사람이 알기 때문.

[미검증] Isaac 에서 한 번도 안 돌려 봤다. 서보 응답 지연, 버스 구성, 고장 서보가 실제로 어떻게
움직이는지 (힘 빠짐 / 굳음) 는 실물에서 확인해야 한다.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp as gait_mdp
from .cpg_action import TripodCPGAction, TripodCPGActionCfg
from .robot_cfg import LEG_JOINTS

NONE, LIMP, LOCKED, WEAK = 0, 1, 2, 3
MODE_NAMES = {"limp": LIMP, "locked": LOCKED, "weak": WEAK}
JOINT_KINDS = ("coxa", "femur", "tibla")
SERVO_TICK = math.radians(240.0 / 1000.0)   # HX-35H 위치 한 눈금 (0.24 도)


# ---------------------------------------------------------------- 공용 상태 (env 에 붙여 둔다)
def _state(env) -> dict:
    """env 마다 (N, 18) 서보 상태. 관절 순서는 gait_mdp.LEGS x (coxa, femur, tibla) = CPG 행동 순서."""
    st = getattr(env, "_servo", None)
    if st is None:
        n, dev = env.num_envs, env.device
        robot: Articulation = env.scene["robot"]
        names = [f"{k}_{leg}_joint" for leg in gait_mdp.LEGS for k in JOINT_KINDS]
        ids = robot.find_joints(names, preserve_order=True)[0]
        act = robot.actuators["legs"]
        a_ids = act.joint_indices
        a_ids = list(range(robot.num_joints)) if isinstance(a_ids, slice) else torch.as_tensor(a_ids).tolist()
        st = {
            "ids": ids,
            "act": act,
            "act_pos": [a_ids.index(i) for i in ids],
            "k0": robot.data.default_joint_stiffness[:, ids].clone(),
            "d0": robot.data.default_joint_damping[:, ids].clone(),
            "e0": robot.data.joint_effort_limits[:, ids].clone(),
            "offset": torch.zeros(n, 18, device=dev),     # 0점 오차: 실제 각 = 명령 + offset, 읽기 = 실제 - offset
            "lag": torch.zeros(n, 1, device=dev),         # 읽기 지연 (제어 주기 비율, 0 ~ 1)
            "mask": torch.zeros(n, 18, dtype=torch.bool, device=dev),
            "mode": torch.zeros(n, 18, dtype=torch.long, device=dev),
            "lock": torch.zeros(n, 18, device=dev),
            "leg": torch.zeros(n, 6, device=dev),         # 고장 다리 one-hot
        }
        env._servo = st
    return st


def parse_fault(spec: str):
    """평가용 고정 고장 'LF', 'LF:locked', 'LF.femur:limp', 'none' -> (다리 번호, 관절 번호들, 모드) 또는 None."""
    if spec in ("", "none"):
        return None
    where, _, mode = spec.partition(":")
    leg, _, kind = where.partition(".")
    if leg not in gait_mdp.LEGS:
        raise SystemExit(f"모르는 다리 '{leg}'. {gait_mdp.LEGS} 중 하나")
    joints = [JOINT_KINDS.index(kind)] if kind else [0, 1, 2]
    return gait_mdp.LEGS.index(leg), joints, MODE_NAMES[mode or "limp"]


def reset_servo_state(
    env,
    env_ids: torch.Tensor,
    p_fault: float = 0.0,
    p_whole_leg: float = 0.5,
    mode_probs: tuple[float, float, float] = (0.45, 0.35, 0.20),
    limp_damping: tuple[float, float] = (0.05, 0.5),
    weak_scale: tuple[float, float] = (0.15, 0.5),
    lock_range: float = 0.3,
    offset_range: float = 0.03,
    max_lag: float = 1.0,
    fixed: str | None = None,
):
    """리셋 이벤트: 0점 오차, 읽기 지연, 고장을 다시 뽑고 고장 관절의 강성 / 감쇠 / 토크를 시뮬에 쓴다.
    fixed 를 주면 (평가) 모든 env 가 그 고장, 0점 오차 0."""
    st = _state(env)
    robot: Articulation = env.scene["robot"]
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.arange(env.num_envs, device=env.device)
    ids = ids.to(env.device, torch.long)
    n, dev = len(ids), env.device
    if n == 0:
        return

    st["offset"][ids] = 0.0 if fixed is not None else (torch.rand(n, 18, device=dev) * 2 - 1) * offset_range
    st["lag"][ids] = torch.rand(n, 1, device=dev) * max_lag

    mask = torch.zeros(n, 18, dtype=torch.bool, device=dev)
    mode = torch.zeros(n, 18, dtype=torch.long, device=dev)
    leg_onehot = torch.zeros(n, 6, device=dev)
    fx = parse_fault(fixed) if fixed is not None else None
    if fixed is not None:
        if fx is not None:
            leg, joints, m = fx
            for j in joints:
                mask[:, leg * 3 + j] = True
            mode[mask] = m
            leg_onehot[:, leg] = 1.0
    elif p_fault > 0:
        faulty = torch.rand(n, device=dev) < p_fault
        leg = torch.randint(0, 6, (n,), device=dev)
        whole = torch.rand(n, device=dev) < p_whole_leg
        one = torch.randint(0, 3, (n,), device=dev)
        k = torch.arange(3, device=dev)
        jmask = whole.unsqueeze(-1) | (k.unsqueeze(0) == one.unsqueeze(-1))           # (n, 3)
        cols = leg.unsqueeze(-1) * 3 + k.unsqueeze(0)                                  # (n, 3)
        mask.scatter_(1, cols, jmask & faulty.unsqueeze(-1))
        m = torch.multinomial(torch.tensor(mode_probs, device=dev), n, replacement=True) + 1   # 다리마다 한 모드
        mode = torch.where(mask, m.unsqueeze(-1), mode)
        leg_onehot[faulty, leg[faulty]] = 1.0
    st["mask"][ids] = mask
    st["mode"][ids] = mode
    st["leg"][ids] = leg_onehot
    st["lock"][ids] = robot.data.default_joint_pos[ids][:, st["ids"]] + (torch.rand(n, 18, device=dev) * 2 - 1) * lock_range

    # 강성 / 감쇠 / 토크: 건강하면 기본값
    lo, hi = limp_damping
    k_ = torch.where(mode == LIMP, 0.0, st["k0"][ids])
    d_ = torch.where(mode == LIMP, lo + (hi - lo) * torch.rand(n, 18, device=dev), st["d0"][ids])
    wlo, whi = weak_scale
    e_ = torch.where(mode == WEAK, st["e0"][ids] * (wlo + (whi - wlo) * torch.rand(n, 18, device=dev)), st["e0"][ids])
    robot.write_joint_stiffness_to_sim(k_, joint_ids=st["ids"], env_ids=ids)
    robot.write_joint_damping_to_sim(d_, joint_ids=st["ids"], env_ids=ids)
    robot.write_joint_effort_limit_to_sim(e_, joint_ids=st["ids"], env_ids=ids)
    # 토크 벌점 (dof_torques_l2) 이 쓰는 액추에이터 추정 토크도 같은 값으로 (randomize_actuator_gains 와 같은 방식)
    act, pos = st["act"], st["act_pos"]
    act.stiffness[ids[:, None], pos] = k_
    act.damping[ids[:, None], pos] = d_


# ---------------------------------------------------------------- 행동: CPG + 서보 흉내
class ServoCPGAction(TripodCPGAction):
    """TripodCPGAction 과 행동 크기 · 뜻이 같다 (3-m 체크포인트를 그대로 이어 받음). 다른 점:
    - 0점 오차: 서보에 보내는 목표 + offset
    - 고장: locked 는 고정 각도, limp 는 강성 0 이라 목표 무관 (현재 각도로 둠)
    - interp: 제어 주기 동안 이전 목표 -> 새 목표 직선 보간 (서보 MOVE_TIME = 제어 주기)"""

    cfg: "ServoCPGActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = torch.zeros(self.num_envs, 18, device=self.device)
        self._cmd = torch.zeros(self.num_envs, 18, device=self.device)
        self._k = 0
        self._decim = max(1, int(env.cfg.decimation))

    def process_actions(self, actions: torch.Tensor):
        self._prev[:] = self._cmd
        super().process_actions(actions)                    # self._targets = 걸음 + 보정 (정책이 생각하는 각도)
        st = _state(self._env)
        cmd = self._targets + st["offset"]
        q = self._asset.data.joint_pos[:, self._joint_ids]
        cmd = torch.where(st["mode"] == LOCKED, st["lock"], cmd)
        cmd = torch.where(st["mode"] == LIMP, q, cmd)
        self._cmd[:] = cmd
        if not self.cfg.interp:
            self._prev[:] = cmd
        self._k = 0

    def apply_actions(self):
        self._k += 1
        w = min(1.0, self._k / self._decim)
        tgt = self._prev + w * (self._cmd - self._prev)
        self._asset.set_joint_position_target(tgt, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        d = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]
        self._targets[env_ids] = d
        self._cmd[env_ids] = d
        self._prev[env_ids] = d


@configclass
class ServoCPGActionCfg(TripodCPGActionCfg):
    class_type: type = ServoCPGAction
    interp: bool = True


def to_servo_action(cfg: TripodCPGActionCfg, interp: bool = True) -> ServoCPGActionCfg:
    """기존 CPG 행동 설정을 그대로 옮긴 ServoCPGActionCfg."""
    import dataclasses  # noqa: PLC0415

    new = ServoCPGActionCfg(interp=interp)
    for f in dataclasses.fields(cfg):
        if f.name != "class_type":
            setattr(new, f.name, getattr(cfg, f.name))
    return new


# ---------------------------------------------------------------- 관측: 서보 읽기값
class ServoReading(ManagerTermBase):
    """관측: 서보가 돌려주는 관절 각 (선 자세 기준, rad) 또는 그 차분 속도 (rad/s).
    params: out = "pos" | "vel", quantize (0.24 도 눈금), mask_faulty (고장 관절 0).
    지연은 env 마다 lag (0 ~ 1 주기): 읽기 = 지금 - lag x (지금 - 직전). 차분 속도는 읽기값끼리 빼서 만든다
    (실물에서 할 수 있는 그대로). 잡음은 ObsTerm noise 로."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        # 출력 순서는 기존 joint_pos / joint_vel 관측(SceneEntityCfg(joint_names=LEG_JOINTS))과 같게 -> 3-m 체크포인트 그대로
        st = _state(env)
        obs_ids = env.scene["robot"].find_joints(cfg.params.get("joint_names", LEG_JOINTS))[0]
        self.perm = [st["ids"].index(i) for i in obs_ids]
        self.prev_q = torch.zeros(n, 18, device=env.device)
        self.prev_read = torch.zeros(n, 18, device=env.device)
        self.fresh = torch.ones(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fresh[env_ids] = True

    def __call__(self, env, out: str = "pos", quantize: bool = True, mask_faulty: bool = True, joint_names=None):
        st = _state(env)
        robot: Articulation = env.scene["robot"]
        q = robot.data.joint_pos[:, st["ids"]]
        f = self.fresh.unsqueeze(-1)
        prev_q = torch.where(f, q, self.prev_q)
        read = q - st["lag"] * (q - prev_q) - st["offset"]
        if quantize:
            read = torch.round(read / SERVO_TICK) * SERVO_TICK
        read = read - robot.data.default_joint_pos[:, st["ids"]]
        if mask_faulty:   # 힘 빠진(죽은) 서보는 위치도 못 준다고 가정. 굳음 / 약함은 읽힘
            read = torch.where(st["mode"] == LIMP, 0.0, read)
        prev_read = torch.where(f, read, self.prev_read)
        self.prev_q = q.clone()
        self.prev_read = read.clone()
        self.fresh[:] = False
        if out == "vel":
            return ((read - prev_read) / env.step_dt)[:, self.perm]
        return read[:, self.perm]


def fault_legs(env) -> torch.Tensor:
    """관측: 고장 다리 one-hot (env, 6). 순서 LF LM LR RF RM RR. 고장 없으면 0."""
    return _state(env)["leg"]


def fault_joints(env) -> torch.Tensor:
    """critic 특권 관측: 관절별 고장 모드 (env, 18 x 3) one-hot (limp, locked, weak)."""
    m = _state(env)["mode"]
    return torch.cat([(m == k).float() for k in (LIMP, LOCKED, WEAK)], dim=-1)


# ---------------------------------------------------------------- 보상: 건강한 다리만 보는 tripod 항목
def _healthy(env) -> torch.Tensor:
    return _state(env)["leg"] < 0.5   # (env, 6)


def ft_tripod_contact(env) -> torch.Tensor:
    _, _, contact = gait_mdp._feet(env)
    ok = (contact == gait_mdp._stance_wanted(env)).float()
    h = _healthy(env).float()
    return (ok * h).sum(-1) / h.sum(-1).clamp(min=1)


def ft_stance_air(env) -> torch.Tensor:
    _, _, contact = gait_mdp._feet(env)
    want = gait_mdp._stance_wanted(env) & _healthy(env)
    return ((~contact) & want).float().sum(-1) / want.float().sum(-1).clamp(min=1)


def ft_stance_slip(env) -> torch.Tensor:
    _, v, contact = gait_mdp._feet(env)
    return (v * contact * gait_mdp._stance_wanted(env) * _healthy(env)).sum(-1)


def ft_swing_clearance(env, target: float = 0.05) -> torch.Tensor:
    z, _, _ = gait_mdp._feet(env)
    lift = (z - z.min(dim=-1, keepdim=True).values).clamp(0.0, target) / target
    swing = ~gait_mdp._stance_wanted(env) & _healthy(env)
    return (lift * swing).sum(-1) / swing.sum(-1).clamp(min=1)


def ft_undesired_contacts(env, threshold: float = 1.0) -> torch.Tensor:
    """벌점용: 건강한 다리의 허벅지(femur)가 땅에 닿은 개수. 힘 빠진 다리가 끌리는 건 정책 탓이 아니라 뺀다."""
    if not hasattr(env, "_femur_sensor_ids"):
        sensor = env.scene["contact_forces"]
        env._femur_sensor_ids = [sensor.find_bodies(f"femur_{leg}")[0][0] for leg in gait_mdp.LEGS]
    f = env.scene["contact_forces"].data.net_forces_w_history[:, :, env._femur_sensor_ids].norm(dim=-1).amax(dim=1)
    return ((f > threshold) & _healthy(env)).float().sum(-1)


# ---------------------------------------------------------------- 설정 도우미
def set_control_hz(env_cfg, hz: float) -> None:
    """물리 스텝은 그대로 (0.005 s), decimation 만 바꿔 정책 주기를 맞춘다. 18 Hz -> 11 (0.055 s, 18.2 Hz)."""
    env_cfg.decimation = max(1, round(1.0 / (hz * env_cfg.sim.dt)))
    env_cfg.sim.render_interval = env_cfg.decimation


def match_discount(env_cfg, agent_cfg, ref_dt: float = 0.02) -> None:
    """50 Hz 에서 쓰던 gamma 0.99 / lambda 0.95 와 같은 '초 단위 시야' 가 되게. 18 Hz -> 0.973 / 0.868."""
    k = env_cfg.sim.dt * env_cfg.decimation / ref_dt
    agent_cfg.algorithm.gamma = 0.99 ** k
    agent_cfg.algorithm.lam = 0.95 ** k


__all__ = [
    "SceneEntityCfg", "ServoCPGActionCfg", "ServoReading", "fault_legs", "fault_joints", "reset_servo_state",
    "set_control_hz", "match_discount", "to_servo_action", "parse_fault",
]
