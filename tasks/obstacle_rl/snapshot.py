#!/usr/bin/env python3
"""학습 맵과 로봇이 제대로 서는지 사진으로 확인. 정책 없이 선 자세(행동 0)로 몇 초 둔다.

    python tasks/obstacle_rl/snapshot.py --headless --enable_cameras

결과: outputs/obstacle_rl/snapshot/
  overview.png           학습 맵 전체
  env<번호>_<칸 종류>.png  열마다 한 마리씩 가까이서
  stand.txt              env 별 몸통 높이, 기울기, 몸통 접촉 여부
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 장애물 맵 스냅샷")
parser.add_argument("--num_envs", type=int, default=14, help="기본 14 = 열마다 한 마리")
parser.add_argument("--settle_s", type=float, default=2.0, help="사진 찍기 전 선 자세로 둘 시간 (s)")
parser.add_argument("--view", action="store_true", help="사진 없이 창을 띄워 두고 선 자세로 계속 둔다. 창을 닫으면 끝")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg  # noqa: E402
from obstacle_rl.terrain_cfg import column_kinds  # noqa: E402

OUT = REPO / "outputs" / "obstacle_rl" / "snapshot"


def shot(env: ManagerBasedRLEnv, eye, target, name: str) -> None:
    env.sim.set_camera_view(eye=list(map(float, eye)), target=list(map(float, target)))
    for _ in range(8):  # RTX 누적이 끝나야 흐리지 않다
        env.sim.render()
    img = env.render()
    if img is None:
        print(f"[snapshot] {name}: 이미지 없음 (--enable_cameras 확인)")
        return
    Image.fromarray(np.asarray(img)).save(OUT / name)
    print(f"[snapshot] {OUT / name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env_cfg = RoSpiderObstacleEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.viewer.resolution = (1280, 720)
    if args.view:
        env_cfg.commands.base_velocity.debug_vis = False  # 화면을 가리는 속도 화살표 끔
        env_cfg.viewer.eye = (-1.0, -1.5, 1.0)
        env_cfg.viewer.lookat = (1.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 10  # 맵 1
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None if args.view else "rgb_array")

    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    if args.view:
        print("[snapshot] 창을 닫으면 끝난다")
        while simulation_app.is_running():
            env.step(act)
        env.close()
        return
    for _ in range(int(args.settle_s / env.step_dt)):
        env.step(act)

    robot = env.scene["robot"]
    terrain = env.scene.terrain
    kinds = column_kinds(env_cfg.scene.terrain.terrain_generator)
    cols = terrain.terrain_types.tolist()
    rows = terrain.terrain_levels.tolist()
    origins = env.scene.env_origins.cpu().numpy()
    root = robot.data.root_pos_w.cpu().numpy()
    grav = robot.data.projected_gravity_b.cpu().numpy()
    tilt = np.degrees(np.arccos(np.clip(-grav[:, 2], -1.0, 1.0)))
    contact = env.scene["contact_forces"]
    base_id = contact.find_bodies("base_footprint")[0]
    base_f = contact.data.net_forces_w[:, base_id].norm(dim=-1).max(dim=-1).values.cpu().numpy()

    lines = ["env  열  행  칸종류     루트z-원점z(m)  기울기(deg)  몸통접촉력(N)"]
    for i in range(env.num_envs):
        lines.append(f"{i:3d} {cols[i]:3d} {rows[i]:3d}  {kinds[cols[i]]:<10} {root[i, 2] - origins[i, 2]:+.3f}"
                     f"          {tilt[i]:5.1f}        {base_f[i]:.2f}")
    (OUT / "stand.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    allo = terrain.terrain_origins.reshape(-1, 3).cpu().numpy()
    c = allo.mean(axis=0)
    span = allo.max(axis=0) - allo.min(axis=0)
    shot(env, c + np.array([-0.6 * span[0] - 4.0, -0.6 * span[1] - 4.0, 0.6 * max(span[:2]) + 6.0]), c, "overview.png")

    seen = set()
    for i in range(env.num_envs):
        if kinds[cols[i]] in seen:
            continue
        seen.add(kinds[cols[i]])
        p = root[i]
        shot(env, p + np.array([-0.5, -1.4, 0.8]), p + np.array([1.0, 0.0, 0.0]), f"env{i:02d}_{kinds[cols[i]]}.png")
        shot(env, p + np.array([-0.25, -0.35, 0.18]), p + np.array([0.05, 0.0, 0.05]), f"env{i:02d}_{kinds[cols[i]]}_close.png")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
