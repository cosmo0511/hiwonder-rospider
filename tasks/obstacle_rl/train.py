#!/usr/bin/env python3
"""장애물 넘기 정책 학습 (RSL-RL PPO). **isaac-shell 에서, 헤드리스로 돌린다.**

    # 처음 한 번: 베이스가 떠 있는 USD
    python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base

    # 학습 (VRAM 8 GB: num_envs 64 부터, 카메라 없음)
    python tasks/obstacle_rl/train.py --headless --num_envs 64

    # 실물 높이에서 안 오르면: 낮은 높이부터 키우는 커리큘럼
    python tasks/obstacle_rl/train.py --headless --num_envs 64 --height_curriculum

    # 이어서
    python tasks/obstacle_rl/train.py --headless --resume logs/rsl_rl/rospider_obstacle/<날짜>/model_1000.pt

로그와 체크포인트: logs/rsl_rl/rospider_obstacle/<날짜_시각>/ (텐서보드: tensorboard --logdir logs/rsl_rl)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 장애물 넘기 PPO 학습")
parser.add_argument("--num_envs", type=int, default=64, help="병렬 env 수. 8 GB VRAM 은 64 부터 두 배씩")
parser.add_argument("--max_iterations", type=int, default=None, help="기본은 agents/rsl_rl_ppo_cfg.py 의 3000")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--height_curriculum", action="store_true",
                    help="난이도(행)에 따라 높이를 0.5 배 -> 1 배로. 기본은 실물 높이 고정")
parser.add_argument("--resume", type=str, default=None, help="이어 학습할 체크포인트 (.pt)")
parser.add_argument("--exp", default=None, help="experiments.py 의 실험 이름 (예: 2c). 주면 --flat/--gait/--blind/--wasd_gains 는 무시")
parser.add_argument("--flat", action="store_true", help="평지에서 보행만 (1단계). 장애물 학습은 이 체크포인트를 --resume")
parser.add_argument("--gait", action="store_true", help="wasd tripod 걸음 모방 + 박자 관측 (1단계)")
parser.add_argument("--blind", action="store_true",
                    help="정책은 IMU + 관절만 본다. 몸 선속도, 높이 스캔은 critic 만 (실기에 없는 값)")
parser.add_argument("--wasd_gains", action="store_true", help="다리 게인을 rospider_wasd.py 값으로 (57.3 / 1.15 / 2.0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl import experiments  # noqa: E402
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg, make_blind, make_flat, use_tripod_gait, use_wasd_gains  # noqa: E402
from obstacle_rl.terrain_cfg import enable_height_curriculum  # noqa: E402


def main() -> None:
    env_cfg = RoSpiderObstacleEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device:
        env_cfg.sim.device = args.device
    agent_cfg = RoSpiderObstaclePPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = env_cfg.sim.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.exp:
        experiments.apply(args.exp, env_cfg, agent_cfg)
    else:
        if args.flat:
            make_flat(env_cfg)
        elif args.height_curriculum:
            enable_height_curriculum(env_cfg.scene.terrain.terrain_generator)
        if args.gait:
            use_tripod_gait(env_cfg)   # make_blind 보다 먼저 (critic 에도 박자)
        if args.blind:
            make_blind(env_cfg)
            agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        if args.wasd_gains:
            use_wasd_gains(env_cfg)

    log_dir = REPO / "logs" / "rsl_rl" / agent_cfg.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + (f"_exp{args.exp}" if args.exp else "".join(f"_{k}" for k in ("flat", "gait", "blind", "wasd_gains") if getattr(args, k))))
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    print(f"[train] 로그: {log_dir}")
    if args.exp:
        print(f"[train] env {args.num_envs}개, 실험 {args.exp}: {experiments.EXPERIMENTS[args.exp].__doc__.strip()}")
    else:
        terrain = "평지" if args.flat else ("커리큘럼 0.5 -> 1.0" if args.height_curriculum else "실물 고정")
        print(f"[train] env {args.num_envs}개, 지형 {terrain}, gait={args.gait}, blind={args.blind}, wasd_gains={args.wasd_gains}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    # 물리 폭발이 한 스텝이라도 생기면 점수가 수백만으로 튀어 critic 이 발산한다 (3-c 4303 회). 한 스텝 점수를 묶어 둔다.
    # 정상 한 스텝 점수는 0.1 안팎, 목표 도착 보너스만 2.0.
    _compute = env.reward_manager.compute
    env.reward_manager.compute = lambda dt: torch.nan_to_num(_compute(dt), nan=0.0, posinf=3.0, neginf=-3.0).clamp_(-3.0, 3.0)
    # 3-c 4407 회: 점수를 묶어도 critic 손실이 매 회 1000 배씩 커짐 -> 관측 어딘가가 비정상. 어느 항목인지 기록하고 ±100 으로 자른다 (관절 속도는 착지 때 40 rad/s 까지 정상).
    om = env.observation_manager
    _cg = om.compute_group
    _warned = {}

    def _checked_group(group_name, update_history=False):
        out = _cg(group_name, update_history=update_history)
        if not isinstance(out, torch.Tensor):
            return out
        bad = ~torch.isfinite(out) | (out.abs() > 100.0)
        if bad.any():
            i0 = 0
            for name, dim in zip(om.active_terms[group_name], om.group_obs_term_dim[group_name]):
                n = int(torch.tensor(dim).prod())
                b = bad[:, i0:i0 + n]
                if b.any() and _warned.get((group_name, name), 0) < 20:
                    _warned[(group_name, name)] = _warned.get((group_name, name), 0) + 1
                    v = out[:, i0:i0 + n][b]
                    print(f"[obs_guard] {group_name}/{name}: env {int(b.any(dim=1).sum())}개, 값 예 {v[:4].tolist()}", flush=True)
                i0 += n
            out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0).clamp_(-100.0, 100.0)
        return out

    om.compute_group = _checked_group
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if args.resume:
        print(f"[train] 이어서: {args.resume}")
        # expand_ckpt.py 로 입력을 늘린 체크포인트는 옵티마이저 상태가 없다 -> 새로 시작
        has_opt = "optimizer_state_dict" in torch.load(args.resume, map_location="cpu", weights_only=False)
        runner.load(args.resume, load_optimizer=has_opt)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
    simulation_app.close()
