#!/usr/bin/env python3
"""학습한 정책을 평지에서 여러 마리 돌리고, **넘어진 장면과 가장 잘 걷는 장면을 자동으로 mp4 로 저장**한다.

    python tasks/obstacle_rl/highlight_eval.py --headless --enable_cameras --exp f18 \
        --checkpoint logs/rsl_rl/rospider_obstacle/<날짜>_expf18/model_XXXX.pt

    # 고장을 하나로 고정해서 (예: 왼쪽 앞 다리 힘 빠짐)
    python tasks/obstacle_rl/highlight_eval.py --headless --enable_cameras --exp "f18@LF:limp" --checkpoint ...

어떻게 고르나
- env 마다 작은 카메라(TiledCamera, 기본 320 x 240) 가 로봇을 옆-뒤에서 따라가며 최근 --clip_s 초를 메모리에 계속 들고 있다.
- 넘어짐 = 시간 초과가 아닌 종료 (몸통이 땅에 닿음 / 1 rad 넘게 기울어짐 / 물리 폭발). 처음 --max_falls 번은
  그 직전 --clip_s 초를 fall_XX.mp4 로 저장.
- 잘 걷기 = 끝날 때 최근 --clip_s 초 동안 안 넘어진 로봇 중 (전진 거리 - 2 x 옆 이동) 이 가장 큰 로봇 -> best_walk.mp4.
- highlights.json 에 각 장면의 env, 시각, 고장 다리 / 모드, 이동 거리를 적는다.

결과: outputs/obstacle_rl/highlights/<이름>/
[미검증] Isaac 에서 아직 안 돌려 봤다. 카메라 16 대 + 정책은 8 GB VRAM 에서 될 것으로 보지만 확인 필요 (--num_envs 를 줄여서 시작).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obstacle_rl.physics_args import add_physics_args  # noqa: E402

parser = argparse.ArgumentParser(description="RosPider 넘어짐 / 잘 걷기 장면 자동 저장")
parser.add_argument("--exp", required=True, help="experiments.py 실험 이름 (예: r18, f18, f18@LF:limp)")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--vx", type=float, default=0.15, help="직진 명령 (m/s)")
parser.add_argument("--friction", type=float, default=0.8)
parser.add_argument("--clip_s", type=float, default=5.0, help="장면 하나 길이 (초)")
parser.add_argument("--max_falls", type=int, default=2)
parser.add_argument("--res", type=int, nargs=2, default=[320, 240], help="카메라 가로 세로")
parser.add_argument("--name", default=None)
add_physics_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl import experiments  # noqa: E402
from obstacle_rl import mdp as gait_mdp  # noqa: E402
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg, make_flat, physics_from_args  # noqa: E402

LEGS = gait_mdp.LEGS
MODES = {0: "-", 1: "limp", 2: "locked", 3: "weak"}


def fault_label(env, i: int) -> str:
    st = getattr(env, "_servo", None)
    if st is None:
        return "고장 없음"
    legs = [LEGS[k] for k in range(6) if st["leg"][i, k] > 0.5]
    if not legs:
        return "고장 없음"
    leg = LEGS.index(legs[0])
    parts = [f"{kind}:{MODES[int(st['mode'][i, leg * 3 + j])]}" for j, kind in enumerate(("coxa", "femur", "tibla"))
             if int(st["mode"][i, leg * 3 + j]) != 0]
    return f"{legs[0]} " + ", ".join(parts)


def save_clip(path: Path, frames: list[np.ndarray], fps: float) -> None:
    with imageio.get_writer(path, fps=max(1, round(fps)), codec="libx264", quality=7) as w:
        for f in frames:
            w.append_data(f)
    imageio.imwrite(path.with_suffix(".png"), frames[len(frames) // 2])


def main() -> None:
    name = args.name or f"{Path(args.checkpoint).resolve().parent.name}_{Path(args.checkpoint).stem}_{args.exp.replace(':', '-')}"
    out = REPO / "outputs" / "obstacle_rl" / "highlights" / name
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = RoSpiderObstacleEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = RoSpiderObstaclePPORunnerCfg()
    experiments.apply(args.exp, env_cfg, agent_cfg)
    make_flat(env_cfg)
    physics_from_args(env_cfg, args)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.vx, args.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    gait_mdp.straight_only(cmd)
    env_cfg.observations.policy.enable_corruption = False
    ev = env_cfg.events
    ev.push_robot = None
    ev.base_external_force_torque = None
    ev.physics_material.params["static_friction_range"] = (args.friction, args.friction)
    ev.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    if getattr(ev, "leg_armature", None) is not None:
        ev.leg_armature = None
    if getattr(ev, "real_masses", None) is not None:
        ev.real_masses.params["scale_range"] = (1.0, 1.0)
    env_cfg.episode_length_s = args.seconds + 5.0
    # env 마다 따라가는 작은 카메라 (자세는 매 스텝 set_world_poses_from_view 로)
    env_cfg.scene.rec_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/RecCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.6, -0.8, 0.5), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        width=args.res[0],
        height=args.res[1],
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

    agent_cfg.device = env_cfg.sim.device
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    pobs = wrapped.get_observations()
    pobs = pobs[0] if isinstance(pobs, tuple) else pobs

    robot = env.scene["robot"]
    cam = env.scene["rec_cam"]
    n = env.num_envs
    fps = 1.0 / env.step_dt
    keep = max(2, int(args.clip_s * fps))
    buf = np.zeros((n, keep, args.res[1], args.res[0], 3), dtype=np.uint8)   # env 마다 최근 clip_s 초 (고리 버퍼)
    count = np.zeros(n, dtype=np.int64)          # 이번 에피소드에 쌓인 프레임 수
    ep_start = robot.data.root_pos_w[:, :2].clone()
    labels = [fault_label(env, i) for i in range(n)]
    falls: list[dict] = []
    off = torch.tensor([-0.45, -0.55, 0.30], device=env.device)
    look = torch.tensor([0.10, 0.0, 0.03], device=env.device)
    steps = int(args.seconds / env.step_dt)
    print(f"[highlight] env {n}, {args.seconds:.0f} s, {fps:.1f} Hz, 장면 {args.clip_s} s", flush=True)

    for t in range(steps):
        root = robot.data.root_pos_w
        cam.set_world_poses_from_view(root + off, root + look)
        pos_before = root[:, :2].clone()
        with torch.inference_mode():
            pobs, _, dones, extras = wrapped.step(policy(pobs))
            pobs = pobs[0] if isinstance(pobs, tuple) else pobs
        # 이번 스텝 렌더는 리셋 전에 찍힌 것 (Isaac Lab 은 물리 루프 안에서 렌더하고 리셋은 그 뒤)
        rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        for i in range(n):
            buf[i, count[i] % keep] = rgb[i]
        count += 1
        fell = (env.termination_manager.terminated & ~env.termination_manager.time_outs).cpu().numpy()
        done = dones.bool().cpu().numpy()
        for i in np.flatnonzero(fell):
            if len(falls) < args.max_falls and count[i] >= fps:   # 출발 직후 1 초 안 넘어짐은 빼고
                k = min(count[i], keep)
                frames = [buf[i, (count[i] - k + j) % keep] for j in range(k)]
                fname = f"fall_{len(falls):02d}.mp4"
                save_clip(out / fname, frames, fps)
                dist = (pos_before[i] - ep_start[i]).cpu().numpy()
                why = [nm for nm in env.termination_manager.active_terms
                       if bool(env.termination_manager.get_term(nm)[i])]
                falls.append({"file": fname, "env": int(i), "t_s": round(float(count[i] / fps), 1),
                              "fault": labels[i], "why": why, "dx_m": round(float(dist[0]), 3), "dy_m": round(float(dist[1]), 3)})
                print(f"[highlight] 넘어짐 저장 {fname}: env {i}, {falls[-1]}", flush=True)
        for i in np.flatnonzero(done):    # 리셋된 env: 버퍼, 출발점, 고장 이름 새로
            count[i] = 0
            ep_start[i] = robot.data.root_pos_w[i, :2]
            labels[i] = fault_label(env, i)

    # 잘 걷기: 최근 clip_s 초 동안 안 끊긴 로봇 중 점수 최고
    d = (robot.data.root_pos_w[:, :2] - ep_start).cpu().numpy()
    score = d[:, 0] - 2.0 * np.abs(d[:, 1])
    ok = count >= keep
    best = None
    if ok.any():
        i = int(np.flatnonzero(ok)[np.argmax(score[ok])])
        frames = [buf[i, (count[i] - keep + j) % keep] for j in range(keep)]
        save_clip(out / "best_walk.mp4", frames, fps)
        best = {"file": "best_walk.mp4", "env": i, "fault": labels[i], "episode_s": round(float(count[i] / fps), 1),
                "dx_m": round(float(d[i, 0]), 3), "dy_m": round(float(d[i, 1]), 3)}
        print(f"[highlight] 잘 걷기 저장: {best}", flush=True)
    else:
        print("[highlight] clip_s 초 넘게 버틴 로봇이 없어서 잘 걷기 장면 없음", flush=True)
    summary = {"exp": args.exp, "checkpoint": str(args.checkpoint), "num_envs": n, "seconds": args.seconds,
               "vx_cmd": args.vx, "control_hz": round(fps, 2), "falls": falls, "best_walk": best,
               "fall_total_note": "falls 에는 처음 max_falls 개만 저장"}
    (out / "highlights.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[highlight] 결과: {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
