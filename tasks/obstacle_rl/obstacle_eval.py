#!/usr/bin/env python3
"""장애물 통과율 평가. scripts/obstacle_sweep.py (rospider-isaacsim 레포) 와 같은 장애물 모양 / 판정으로
RL 정책과 기존 tripod 걸음(베이스라인)을 같은 Isaac Lab 환경에서 비교한다.

    python tasks/obstacle_rl/obstacle_eval.py --headless --controller tripod
    python tasks/obstacle_rl/obstacle_eval.py --headless --controller policy --checkpoint <...>.pt --exp 3a
    # 마찰 0.5:  --friction 0.5    케이스 바꾸기: --cases bump:0.08:0.40 pit:0.05:0.25

장애물 (obstacle_sweep 과 같음): 로봇 출발점에서 0.70 m 앞이 장애물 중심, 레인 폭 1.2 m.
- 방지턱 bump:높이:폭 = 원호 단면, 폭은 진행 방향 길이
- 구덩이 pit:깊이:길이 = 바닥 전체를 깊이만큼 올리고 그 구간만 비움 (상자 모양)
판정 (obstacle_sweep 과 같게): 장애물 끝 + 0.35 m 를 35 s 안에 지나면 통과 (몸통이 닿았어도 통과, 따로 셈).
실패 = 뒤집힘 / 레인 이탈(|y| > 0.6 m) / 시간 초과. 베이스라인은 시작 박자를 무작위로, 출발 전 1.5 s 서 있음.
결과: outputs/obstacle_rl/obstacle_eval/<이름>/results.csv, summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RoSpider 장애물 통과율")
parser.add_argument("--controller", choices=["policy", "tripod"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--exp", default="2d", help="정책을 학습한 실험 이름 (관측 크기를 맞추려고). tripod 는 무엇이든 됨")
parser.add_argument("--cases", nargs="+",
                    default=["bump:0.08:0.40", "bump:0.08:0.50", "bump:0.08:0.60", "pit:0.05:0.25"])
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--friction", type=float, default=0.8)
parser.add_argument("--vx", type=float, default=0.18, help="정책 전진 명령 (m/s)")
parser.add_argument("--name", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

from obstacle_rl import course, experiments  # noqa: E402
from obstacle_rl import mdp  # noqa: E402
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg, use_wasd_gains  # noqa: E402
from obstacle_rl.baseline import TripodBaseline  # noqa: E402

OBST_X, LANE_HALF, PASS_MARGIN, TIMEOUT_S, FLIP_UP_Z = 0.70, 0.60, 0.35, 35.0, 0.5
START_X = course.START_LINE + course.ROBOT_FROM_START   # 타일 안 로봇 출발 x


def bump_profile(x, h, w):
    """obstacle_sweep.bump_profile 과 같음 (원호 단면)."""
    if x <= 0.0 or x >= w:
        return 0.0
    r = (w * w / 4.0 + h * h) / (2.0 * h)
    d = x - w / 2.0
    return max(0.0, math.sqrt(max(0.0, r * r - d * d)) - (r - h))


def sweep_tile(difficulty: float, cfg: "SweepTileCfg"):
    L, W = course.TILE
    c = START_X + OBST_X
    parts = []
    if cfg.kind == "bump":
        parts.append(course.box(0.0, L, 0.0, W, -course.FLOOR_T, 0.0))
        xs = np.linspace(c - cfg.w / 2, c + cfg.w / 2, 61)
        zt = np.array([max(bump_profile(x - (c - cfg.w / 2), cfg.h, cfg.w), 1e-3) for x in xs])
        parts.append(course.prism(xs, zt, 0.0, 0.0, W))
        origin = np.array([START_X, W / 2, 0.0])
    else:   # pit: 바닥 전체를 h 만큼 올리고 [c - w/2, c + w/2] 만 비운다
        a, b = c - cfg.w / 2, c + cfg.w / 2
        parts.append(course.box(0.0, L, 0.0, W, -course.FLOOR_T, 0.0))
        parts.append(course.box(0.0, a, 0.0, W, 0.0, cfg.h))
        parts.append(course.box(b, L, 0.0, W, 0.0, cfg.h))
        origin = np.array([START_X, W / 2, cfg.h])
    meshes = [trimesh.Trimesh(vertices=V, faces=F, process=False) for V, F in parts]
    return meshes, origin


@configclass
class SweepTileCfg(SubTerrainBaseCfg):
    function = sweep_tile
    kind: str = "bump"
    h: float = 0.08
    w: float = 0.40


def parse_case(s: str) -> dict:
    kind, h, w = s.split(":")
    return {"kind": kind, "h": float(h), "w": float(w), "name": f"{kind}_h{float(h) * 100:.0f}cm_w{float(w) * 100:.0f}cm"}


def main() -> None:
    cases = [parse_case(s) for s in args.cases]
    name = args.name or (f"{args.controller}" + ("" if args.controller == "tripod" else f"_exp{args.exp}")
                         + f"_f{args.friction:g}")
    out = REPO / "outputs" / "obstacle_rl" / "obstacle_eval" / name
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = RoSpiderObstacleEnvCfg()
    agent_cfg = RoSpiderObstaclePPORunnerCfg()
    experiments.apply(args.exp, env_cfg, agent_cfg)
    if args.controller == "tripod":
        use_wasd_gains(env_cfg)
    n_case = len(cases)
    env_cfg.scene.num_envs = n_case * args.trials
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=course.TILE, border_width=1.0, num_rows=1, num_cols=n_case,
        horizontal_scale=0.01, vertical_scale=0.001, slope_threshold=0.75, use_cache=False,
        curriculum=True, difficulty_range=(1.0, 1.0),
        sub_terrains={c["name"]: SweepTileCfg(kind=c["kind"], h=c["h"], w=c["w"], proportion=1.0 / n_case) for c in cases},
    )
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.physics_material.params["static_friction_range"] = (args.friction, args.friction)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    ev = env_cfg.events
    if ev.add_base_mass is not None:
        ev.add_base_mass.params["mass_distribution_params"] = (1.54, 1.54)
    if getattr(ev, "real_masses", None) is not None:
        ev.real_masses.params["scale_range"] = (1.0, 1.0)
    # obstacle_sweep 처럼 출발 위치 / 방향을 조금씩 흔든다
    ev.reset_base.params["pose_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
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
    env_cfg.episode_length_s = TIMEOUT_S + 5.0
    env_cfg.terminations.base_contact = None    # obstacle_sweep 처럼 몸통 접촉으로 끝내지 않음 (기록만)
    if getattr(env_cfg.terminations, "reached_goal", None) is not None:   # 3단계 실험: 도착 순간 리셋되면 판정을 못 함
        env_cfg.terminations.reached_goal = None
    env_cfg.events.reset_pipes = EventTerm(func=mdp.park_pipes, mode="reset")

    env = ManagerBasedRLEnv(cfg=env_cfg)
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
        with torch.inference_mode():
            for _ in range(int(1.5 / env.step_dt)):   # obstacle_sweep SETTLE_S: 출발 전 서 있기
                env.step(tripod.act(vx=0.0, wz=0.0))

    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    base_id = sensor.find_bodies("base_footprint")[0]
    touched = np.zeros(env.num_envs, dtype=bool)
    n = env.num_envs
    col = env.scene.terrain.terrain_types.cpu().numpy()
    origins = env.scene.env_origins.cpu().numpy()
    start = robot.data.root_pos_w.cpu().numpy().copy()
    goal = np.array([OBST_X + cases[c]["w"] / 2 + PASS_MARGIN for c in col])
    result = np.array([""] * n, dtype=object)
    t_done = np.zeros(n)
    max_x = np.zeros(n)
    max_tilt = np.zeros(n)
    steps = int(TIMEOUT_S / env.step_dt)
    for t in range(steps):
        with torch.inference_mode():
            if args.controller == "policy":
                obs, _, dones, _ = wrapped.step(policy(obs))
                obs = obs[0] if isinstance(obs, tuple) else obs
            else:
                _, _, term, trunc, _ = env.step(tripod.act(vx=1.0, wz=0.0))
                dones = term | trunc
        pos = robot.data.root_pos_w.cpu().numpy()
        up_z = -robot.data.projected_gravity_b[:, 2].cpu().numpy()
        dx = pos[:, 0] - start[:, 0]
        dy = pos[:, 1] - origins[:, 1]
        open_ = result == ""
        max_x = np.where(open_, np.maximum(max_x, dx), max_x)
        max_tilt = np.where(open_, np.maximum(max_tilt, np.degrees(np.arccos(np.clip(up_z, -1, 1)))), max_tilt)
        d = dones.cpu().numpy().astype(bool)
        touched |= (sensor.data.net_forces_w[:, base_id].norm(dim=-1).max(dim=-1).values > 1.0).cpu().numpy() & open_
        sec = (t + 1) * env.step_dt
        for i in np.flatnonzero(open_):
            if d[i]:
                result[i] = "실패:종료"
            elif up_z[i] < FLIP_UP_Z:
                result[i] = "실패:뒤집힘"
            elif abs(dy[i]) > LANE_HALF:
                result[i] = "실패:레인이탈"
            elif dx[i] >= goal[i]:
                result[i] = "통과"
            else:
                continue
            t_done[i] = sec
        if not (result == "").any():
            break
    result[result == ""] = "실패:시간초과"

    rows = []
    for i in range(n):
        c = cases[col[i]]
        rows.append({"case": c["name"], "result": result[i], "time_s": round(float(t_done[i]), 2),
                     "max_progress_m": round(float(max_x[i]), 3), "goal_m": round(float(goal[i]), 3),
                     "max_tilt_deg": round(float(max_tilt[i]), 1), "body_contact": bool(touched[i])})
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"[{name}] controller={args.controller} 마찰 {args.friction} 시도 {args.trials}회/케이스"]
    for c in cases:
        rs = [r for r in rows if r["case"] == c["name"]]
        ok = sum(r["result"] == "통과" for r in rs)
        fails = {}
        for r in rs:
            if r["result"] != "통과":
                fails[r["result"]] = fails.get(r["result"], 0) + 1
        ft = ", ".join(f"{k.split(':')[1]} {v}" for k, v in fails.items())
        bc = sum(r["result"] == "통과" and r["body_contact"] for r in rs)
        lines.append(f"  {c['name']:<20} {ok}/{len(rs)} 통과 ({ok / len(rs) * 100:.0f} %, 그중 몸통 닿음 {bc})  {ft}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
