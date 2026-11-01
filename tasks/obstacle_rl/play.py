#!/usr/bin/env python3
"""학습한 정책을 **실물 코스(맵 1, 맵 2)** 에서 평가한다. 학습 지형과 다른 지형이다.

    python tasks/obstacle_rl/play.py --checkpoint logs/rsl_rl/rospider_obstacle/<날짜>/model_3000.pt
    python tasks/obstacle_rl/play.py --checkpoint ... --map 1 --episodes 5          # 맵 1 만
    python tasks/obstacle_rl/play.py --checkpoint ... --headless --episodes 20      # 숫자만

맵마다 에피소드를 돌려 "작업대(높이 0.24)에 올라섰나"와 "출발점에서 얼마나 나아갔나"를 센다.
결과: outputs/obstacle_rl/play_<날짜>.csv
명령은 0.1 m/s 직진 고정, 무작위화와 관측 잡음은 끈다 (env_cfg.RoSpiderObstacleEnvCfg_PLAY).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 장애물 정책 평가 (실물 코스)")
parser.add_argument("--checkpoint", required=True, help="model_XXXX.pt")
parser.add_argument("--map", choices=["1", "2", "both"], default="both")
parser.add_argument("--episodes", type=int, default=3, help="맵마다 에피소드 수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl import course  # noqa: E402
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg_PLAY  # noqa: E402
from obstacle_rl.terrain_cfg import column_kinds  # noqa: E402


def main() -> None:
    env_cfg = RoSpiderObstacleEnvCfg_PLAY()
    gen = env_cfg.scene.terrain.terrain_generator
    if args.map != "both":
        keep = f"map{args.map}"
        gen.sub_terrains = {keep: gen.sub_terrains[keep]}
        gen.num_cols = 1
    env_cfg.scene.num_envs = gen.num_cols   # 맵마다 로봇 하나
    if args.device:
        env_cfg.sim.device = args.device

    agent_cfg = RoSpiderObstaclePPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    kinds_by_col = column_kinds(gen)
    kinds = [kinds_by_col[int(c)] for c in env.scene.terrain.terrain_types.tolist()]
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    goal = {k: course.layout(k)["plat"] - course.origin(k)[0] for k in set(kinds)}
    robot = env.scene["robot"]
    n = env.num_envs
    best = torch.zeros(n, device=env.device)
    done_count = [0] * n
    rows = []

    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    t0 = env.episode_length_buf.clone()
    while simulation_app.is_running() and min(done_count) < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]
        prog = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        best = torch.where(dones.bool(), best, torch.maximum(best, prog))
        reached = env.termination_manager.get_term("platform")
        for i in torch.nonzero(dones).flatten().tolist():
            if done_count[i] >= args.episodes:
                continue
            rows.append({
                "map": kinds[i], "episode": done_count[i] + 1,
                "reached_platform": bool(reached[i]),
                "best_progress_m": round(float(best[i]), 3),
                "goal_m": round(goal[kinds[i]], 3),
                "time_s": round(float(t0[i] + 1) * env.step_dt, 1),
            })
            print(f"[play] {kinds[i]} #{done_count[i] + 1}: 작업대 {'도착' if reached[i] else '실패'}, "
                  f"최대 진척 {float(best[i]):.2f} / {goal[kinds[i]]:.2f} m")
            done_count[i] += 1
            best[i] = 0.0
        t0 = env.episode_length_buf.clone()

    out = REPO / "outputs" / "obstacle_rl"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"play_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["map"])
        w.writeheader()
        w.writerows(rows)
    for k in sorted(set(kinds)):
        rs = [r for r in rows if r["map"] == k]
        if rs:
            print(f"[play] {k}: 성공 {sum(r['reached_platform'] for r in rs)}/{len(rs)}")
    print(f"[play] 결과: {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
