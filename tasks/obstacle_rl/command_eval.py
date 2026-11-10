#!/usr/bin/env python3
"""정지 · 회전 명령을 잘 따르는지 잰다 (3-k). 평지에서 명령을 구간별로 바꿔 가며 속도 / 회전 / 멈춤을 기록한다.

    # 학습한 정책
    python tasks/obstacle_rl/command_eval.py --headless --controller policy --exp 3k \
        --checkpoint logs/rsl_rl/rospider_obstacle/<날짜>_exp3k/model_XXXX.pt

    # 기존 걸음 (wasd 키). wz 는 A 키 1 = 1. 3-k CPG 의 base_yaw_rate 를 여기서 잰다
    python tasks/obstacle_rl/command_eval.py --headless --controller tripod

    # 영상: --video --enable_cameras

구간 (기본): 직진 0.18 -> 정지 -> 제자리 왼쪽 0.4 -> 제자리 오른쪽 0.4 -> 걸으며 왼쪽 (0.16, 0.25) -> 정지.
각 구간 앞 1.5 초는 바뀌는 중이라 지표에서 뺀다.
결과: outputs/obstacle_rl/command_eval/<이름>/metrics.txt (+ walk.mp4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 정지 · 회전 명령 평가")
parser.add_argument("--controller", choices=["policy", "tripod", "zero"], required=True,
                    help="zero = 행동 0 (CPG 만, --exp 필요)")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--exp", default="3k")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--friction", type=float, default=0.8)
parser.add_argument("--video", action="store_true")
parser.add_argument("--name", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.controller == "policy" and not args.checkpoint:
    parser.error("--controller policy 는 --checkpoint 가 필요하다")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl import experiments  # noqa: E402
from obstacle_rl import mdp as gait_mdp  # noqa: E402
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.baseline import TripodBaseline  # noqa: E402
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg, make_flat, use_wasd_gains  # noqa: E402

# (이름, 모드, vx [m/s], wz [rad/s], 초). tripod 는 vx / wz 를 키 값(0 ~ 1)으로 읽는다
SEGMENTS = [
    ("직진", gait_mdp.STRAIGHT, 0.18, 0.0, 6.0),
    ("정지", gait_mdp.STOP, 0.0, 0.0, 5.0),
    ("제자리 왼쪽", gait_mdp.TURN, 0.0, 0.4, 6.0),
    ("제자리 오른쪽", gait_mdp.TURN, 0.0, -0.4, 6.0),
    ("걸으며 왼쪽", gait_mdp.WALK_TURN, 0.16, 0.25, 6.0),
    ("정지", gait_mdp.STOP, 0.0, 0.0, 5.0),
]
TRIPOD_SEGMENTS = [
    ("W", None, 1.0, 0.0, 6.0),
    ("정지", None, 0.0, 0.0, 5.0),
    ("A (제자리 왼쪽)", None, 0.0, 1.0, 6.0),
    ("D (제자리 오른쪽)", None, 0.0, -1.0, 6.0),
    ("W + A", None, 1.0, 1.0, 6.0),
    ("정지", None, 0.0, 0.0, 5.0),
]
SETTLE = 1.5


def main() -> None:
    name = args.name or ("tripod" if args.controller == "tripod" else f"{args.controller}_exp{args.exp}")
    out = REPO / "outputs" / "obstacle_rl" / "command_eval" / name
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = RoSpiderObstacleEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = RoSpiderObstaclePPORunnerCfg()
    if args.controller == "tripod":   # 기존 걸음은 관절 행동 환경 그대로 (gait_eval 과 같음)
        use_wasd_gains(env_cfg)
    else:
        experiments.apply(args.exp, env_cfg, agent_cfg)
    make_flat(env_cfg)
    cmd = env_cfg.commands.base_velocity
    if not isinstance(cmd, gait_mdp.MixedVelocityCommandCfg):
        cmd = gait_mdp.use_mixed_commands(cmd)
        env_cfg.commands.base_velocity = cmd
    cmd.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    ev = env_cfg.events
    ev.push_robot = None
    ev.base_external_force_torque = None
    ev.physics_material.params["static_friction_range"] = (args.friction, args.friction)
    ev.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    if ev.add_base_mass is not None:
        ev.add_base_mass.params["mass_distribution_params"] = (1.54, 1.54)
    if getattr(ev, "leg_armature", None) is not None:   # 3-l: 로봇마다 armature 다르게 하던 것 끔 (평가는 기본 0.01)
        ev.leg_armature = None
    if getattr(ev, "real_masses", None) is not None:
        ev.real_masses.params["scale_range"] = (1.0, 1.0)
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    segs = TRIPOD_SEGMENTS if args.controller == "tripod" else SEGMENTS
    env_cfg.episode_length_s = sum(s[4] for s in segs) + 10.0
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-0.6, -0.8, 0.5)
    env_cfg.viewer.lookat = (0.05, 0.0, 0.05)
    env_cfg.viewer.resolution = (1280, 720)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    env.reset()
    term = env.command_manager.get_term("base_velocity")
    if args.controller == "policy":
        from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

        agent_cfg.device = env_cfg.sim.device
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        pobs = wrapped.get_observations()
        pobs = pobs[0] if isinstance(pobs, tuple) else pobs
    elif args.controller == "tripod":
        tripod = TripodBaseline(env)

    robot = env.scene["robot"]
    n = env.num_envs
    writer = None
    if args.video:
        import imageio  # noqa: PLC0415

        every = max(1, round(1 / (25 * env.step_dt)))
        writer = imageio.get_writer(out / "walk.mp4", fps=round(1 / (every * env.step_dt)), codec="libx264", quality=7)

    lines = [f"[{name}] controller={args.controller}  env {n}개  마찰 {args.friction}  평지",
             f"  각 구간 앞 {SETTLE} 초 빼고 평균. 정지 구간의 '움직인 거리' / '돈 각도' 는 그 구간 처음과 끝 차이"]
    lines.append(f"  {'구간':<14} {'명령 vx':>7} {'명령 wz':>7} | {'vx m/s':>14} {'wz rad/s':>14} {'|vy| m/s':>9} | {'움직인 cm':>9} {'돈 도':>7}")
    was_reset = np.zeros(n, dtype=bool)
    prev_len = env.episode_length_buf.clone()
    t_global = 0
    for seg_name, mode, vx, wz, secs in segs:
        if args.controller != "tripod":
            term.set_fixed(mode, vx, wz)
        steps = int(secs / env.step_dt)
        settle = int(SETTLE / env.step_dt)
        vxs, wzs, vys = [], [], []
        p0 = h0 = None
        for t in range(steps):
            with torch.inference_mode():
                if args.controller == "policy":
                    pobs, _, _, _ = wrapped.step(policy(pobs))
                    pobs = pobs[0] if isinstance(pobs, tuple) else pobs
                elif args.controller == "tripod":
                    env.step(tripod.act(vx=vx, wz=wz))
                else:
                    env.step(torch.zeros(n, env.action_manager.total_action_dim, device=env.device))
            if t == settle:
                p0 = robot.data.root_pos_w[:, :2].clone()
                h0 = robot.data.heading_w.clone()
            if t >= settle:
                vxs.append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
                vys.append(robot.data.root_lin_vel_b[:, 1].abs().cpu().numpy())
                wzs.append(robot.data.root_ang_vel_b[:, 2].cpu().numpy())
            was_reset |= (env.episode_length_buf < prev_len).cpu().numpy()
            prev_len = env.episode_length_buf.clone()
            if writer is not None and t_global % every == 0:
                writer.append_data(env.render())
            t_global += 1
        moved = (robot.data.root_pos_w[:, :2] - p0).norm(dim=-1).cpu().numpy() * 100
        turned = np.degrees(torch.atan2(torch.sin(robot.data.heading_w - h0), torch.cos(robot.data.heading_w - h0)).cpu().numpy())
        ok = ~was_reset
        f = lambda a: f"{a[ok].mean():+.3f} ({a[ok].std():.3f})"  # noqa: E731
        vx_m, wz_m, vy_m = np.mean(vxs, 0), np.mean(wzs, 0), np.mean(vys, 0)
        lines.append(f"  {seg_name:<14} {vx:>7.2f} {wz:>7.2f} | {f(vx_m):>14} {f(wz_m):>14} {vy_m[ok].mean():>9.3f} | "
                     f"{moved[ok].mean():>9.1f} {turned[ok].mean():>+7.1f}")
    lines.append(f"  리셋(넘어짐 등) env: {int(was_reset.sum())}/{n} (지표에서 뺌)")
    if writer is not None:
        writer.close()
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "metrics.txt").write_text(text + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
