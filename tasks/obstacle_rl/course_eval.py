#!/usr/bin/env python3
"""실물 코스 맵 1, 맵 2 평가. 출발선에서 작업대(높이 0.24 m)까지 가는지, 못 가면 어디까지 갔는지 잰다.

    python tasks/obstacle_rl/course_eval.py --headless --controller policy --exp 3a --checkpoint <...>.pt
    python tasks/obstacle_rl/course_eval.py --headless --controller tripod --exp 2c_mass   # 베이스라인 (joint_pos 행동 실험이어야 함)
    # 영상: --video --enable_cameras (맵마다 0 번 로봇을 따라감)

맵 1: 구덩이 -> 파이프 2개(굴러감) -> 그레이팅 계단 4단 -> 작업대. 맵 2: 방지턱 -> 파이프 -> 계단 -> 작업대.
명령은 --vx 직진 + 방향 유지, 밀기 / 관측 잡음 없음, 실측 무게 분포, 마찰 0.8.
결과: outputs/obstacle_rl/course_eval/<이름>/results.csv, summary.txt, map1.mp4, map2.mp4
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RoSpider 실물 코스 평가")
parser.add_argument("--controller", choices=["policy", "tripod"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--exp", default="2f")
parser.add_argument("--trials", type=int, default=10, help="맵마다 시도 수")
parser.add_argument("--vx", type=float, default=0.18)
parser.add_argument("--friction", type=float, default=0.8)
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--name", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl import course, experiments  # noqa: E402
from obstacle_rl import mdp as obs_mdp  # noqa: E402
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.baseline import TripodBaseline  # noqa: E402
from obstacle_rl.env_cfg import use_wasd_gains  # noqa: E402
from obstacle_rl.terrain_cfg import REAL_COURSE_TERRAINS_CFG, column_kinds  # noqa: E402

PLAT_Z = 0.20       # 몸통 루트가 이 높이를 넘으면 작업대 위 (작업대 0.24 m)
LANE_HALF = 0.55    # 코스 폭 1.1 m


def main() -> None:
    name = args.name or (f"{args.controller}" + (f"_exp{args.exp}_{Path(args.checkpoint).stem}" if args.checkpoint else ""))
    out = REPO / "outputs" / "obstacle_rl" / "course_eval" / name
    out.mkdir(parents=True, exist_ok=True)

    env_cfg, agent_cfg = experiments_env(args.exp)
    gen = copy.deepcopy(REAL_COURSE_TERRAINS_CFG)
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.scene.num_envs = 2 * args.trials
    env_cfg.events.reset_pipes = EventTerm(func=obs_mdp.reset_pipes, mode="reset", params={"jitter_x": 0.0})
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.physics_material.params["static_friction_range"] = (args.friction, args.friction)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    ev = env_cfg.events
    if ev.add_base_mass is not None:
        ev.add_base_mass.params["mass_distribution_params"] = (1.54, 1.54)
    if getattr(ev, "real_masses", None) is not None:
        ev.real_masses.params["scale_range"] = (1.0, 1.0)
    ev.reset_base.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.03, 0.03), "yaw": (-0.03, 0.03)}
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.vx, args.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)
    cmd.heading_command = True
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args.seconds + 5.0
    env_cfg.terminations.base_contact = None    # 계단에서 몸통이 스쳐도 계속 가게 둔다 (뒤집힘만 실패)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-0.7, -1.0, 0.6)
    env_cfg.viewer.lookat = (0.2, 0.0, 0.05)
    env_cfg.viewer.resolution = (1280, 720)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    env.reset()
    if args.controller == "policy":
        from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

        agent_cfg.device = env_cfg.sim.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        obs = wrapped.get_observations()
        obs = obs[0] if isinstance(obs, tuple) else obs
    else:
        tripod = TripodBaseline(env, random_phase=True)

    robot = env.scene["robot"]
    n = env.num_envs
    kinds_by_col = column_kinds(gen)
    kinds = [kinds_by_col[int(c)] for c in env.scene.terrain.terrain_types.tolist()]
    origins = env.scene.env_origins.cpu().numpy()
    start = robot.data.root_pos_w.cpu().numpy().copy()
    goal = np.array([course.layout(k)["plat"] - course.origin(k)[0] for k in kinds])
    marks = {k: {o: course.layout(k)[o] - course.origin(k)[0] for o in course.layout(k)} for k in set(kinds)}
    result = np.array([""] * n, dtype=object)
    t_done = np.zeros(n)
    best = np.zeros(n)
    steps = int(args.seconds / env.step_dt)
    writers = {}
    if args.video:
        import imageio  # noqa: PLC0415

        every = max(1, round(1 / (25 * env.step_dt)))
        first = {k: kinds.index(k) for k in sorted(set(kinds))}
    for t in range(steps):
        with torch.inference_mode():
            if args.controller == "policy":
                obs, _, dones, _ = wrapped.step(policy(obs))
                obs = obs[0] if isinstance(obs, tuple) else obs
            else:
                _, _, term, trunc, _ = env.step(tripod.act(vx=1.0, wz=0.0))
                dones = term | trunc
        pos = robot.data.root_pos_w.cpu().numpy()
        dx = pos[:, 0] - start[:, 0]
        dy = pos[:, 1] - origins[:, 1]
        z = pos[:, 2] - origins[:, 2]
        up_z = -robot.data.projected_gravity_b[:, 2].cpu().numpy()
        open_ = result == ""
        best = np.where(open_, np.maximum(best, dx), best)
        d = dones.cpu().numpy().astype(bool)
        for i in np.flatnonzero(open_):
            if z[i] > PLAT_Z and dx[i] > goal[i]:
                result[i] = "완주"
            elif d[i] or up_z[i] < 0.5:
                result[i] = "실패:뒤집힘"
            elif abs(dy[i]) > LANE_HALF:
                result[i] = "실패:코스이탈"
            else:
                continue
            t_done[i] = (t + 1) * env.step_dt
        if args.video and t % every == 0:
            for k, i in first.items():
                env.viewport_camera_controller.set_view_env_index(i)
                for _ in range(2):
                    env.sim.render()
                if k not in writers:
                    writers[k] = imageio.get_writer(out / f"{k}.mp4", fps=round(1 / (every * env.step_dt)),
                                                    codec="libx264", quality=7)
                writers[k].append_data(np.asarray(env.render()))
        if not (result == "").any() and not args.video:
            break
    for w in writers.values():
        w.close()
    result[result == ""] = "실패:시간초과"

    def where(k, x):
        """가장 멀리 간 곳이 어느 장애물 앞 / 위인지."""
        last = "출발 구역"
        for o, xo in sorted(marks[k].items(), key=lambda kv: kv[1]):
            if x >= xo - 0.05:
                last = {"pit": "구덩이", "bump": "방지턱", "pipes": "파이프", "stairs": "계단", "plat": "작업대"}[o]
        return last

    rows = [{"map": kinds[i], "result": result[i], "time_s": round(float(t_done[i]), 1),
             "best_progress_m": round(float(best[i]), 3), "goal_m": round(float(goal[i]), 3),
             "reached": where(kinds[i], best[i])} for i in range(n)]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"[{name}] controller={args.controller} 명령 {args.vx} m/s, 맵마다 {args.trials}회, 제한 {args.seconds:.0f} s"]
    for k in sorted(set(kinds)):
        rs = [r for r in rows if r["map"] == k]
        ok = sum(r["result"] == "완주" for r in rs)
        reach = {}
        for r in rs:
            reach[r["reached"]] = reach.get(r["reached"], 0) + 1
        lines.append(f"  {k}: 완주 {ok}/{len(rs)}, 평균 진척 {np.mean([r['best_progress_m'] for r in rs]):.2f} / "
                     f"{rs[0]['goal_m']:.2f} m, 가장 멀리 간 곳 {reach}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    env.close()


def experiments_env(exp: str):
    from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg  # noqa: PLC0415

    env_cfg = RoSpiderObstacleEnvCfg()
    agent_cfg = RoSpiderObstaclePPORunnerCfg()
    experiments.apply(exp, env_cfg, agent_cfg)
    if args.controller == "tripod":
        use_wasd_gains(env_cfg)
    return env_cfg, agent_cfg


if __name__ == "__main__":
    main()
    simulation_app.close()
