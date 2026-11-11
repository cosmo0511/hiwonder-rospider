"""평가 전용: 다리 서보 명령 지연. 실물은 명령을 보내고 서보가 움직이기 시작할 때까지 50 ~ 150 ms 늦다
(로봇의 rl_walk.py 주석, 2026-11-10 실측). 시뮬엔 이 지연이 없으므로 CPG 행동이 쓰는 관절 목표를
물리 스텝 단위로 늦춰서 넣는다. 순수 지연만 흉내 내고 서보 내부 제어 차이는 넣지 않는다."""

import torch


def add_servo_delay(env, delay_ms: float, term_name: str = "cpg") -> int:
    """CPG 행동 term 의 apply_actions 를 감싸서 관절 목표를 delay_ms 만큼 늦춘다. 늦춘 물리 스텝 수를 돌려준다."""
    k = round(delay_ms / 1000.0 / env.physics_dt)
    if k <= 0:
        return 0
    term = env.action_manager.get_term(term_name)
    buf = term._targets.unsqueeze(0).repeat(k, 1, 1)        # 오래된 것 -> 최신
    refill = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    state = {"i": 0}
    orig_reset = term.reset

    def reset(env_ids=None):
        orig_reset(env_ids)
        if env_ids is None:
            refill[:] = True
        else:
            refill[env_ids] = True

    def apply_actions():
        new = term._targets
        if refill.any():
            buf[:, refill] = new[refill].unsqueeze(0)
            refill[:] = False
        i = state["i"]
        out = buf[i].clone()          # k 스텝 전에 들어온 목표
        buf[i] = new                  # 그 자리에 지금 목표를 넣음
        state["i"] = (i + 1) % k
        term._asset.set_joint_position_target(out, joint_ids=term._joint_ids)

    term.reset = reset
    term.apply_actions = apply_actions
    return k
