#!/usr/bin/env python3
"""공사장 험지를 생성만 해서 위에서 본 높이 지도 PNG 와 메시(.ply)로 저장한다.

물리도 로봇도 없이 지형 생성기만 돈다. `construction_terrain.py` 숫자를 바꾼 뒤
학습 전에 이걸로 먼저 눈으로 확인할 것. isaac-shell 에서 돌린다.

실행:
    python tasks/construction_rl/preview_terrain.py
    python tasks/construction_rl/preview_terrain.py --easy 0.5    # 쉬운 버전

결과: outputs/construction_terrain/{terrain.ply, heightmap.png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="공사장 험지 미리보기")
parser.add_argument("--easy", type=float, default=None, help="easy_variant(scale) 로 생성")
parser.add_argument("--grid", type=float, default=0.04, help="높이 지도 격자 [m]. 작을수록 느리다")
parser.add_argument("--out", type=str, default="outputs/construction_terrain")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

# ---- Kit 이 뜬 뒤 ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.terrains import TerrainGenerator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from construction_terrain import CONSTRUCTION_TERRAINS_CFG, easy_variant  # noqa: E402


def heightmap(mesh, grid: float):
    """위에서 아래로 광선을 쏴서 격자별 최고 높이를 구한다."""
    lo, hi = mesh.bounds
    xs = np.arange(lo[0], hi[0], grid)
    ys = np.arange(lo[1], hi[1], grid)
    gx, gy = np.meshgrid(xs, ys)
    origins = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, hi[2] + 1.0)], axis=1)
    dirs = np.tile([0.0, 0.0, -1.0], (gx.size, 1))
    locs, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    z = np.full(gx.size, np.nan)
    z[idx_ray] = locs[:, 2]
    return xs, ys, z.reshape(gx.shape)


def main():
    cfg = easy_variant(args_cli.easy) if args_cli.easy else CONSTRUCTION_TERRAINS_CFG.copy()
    cfg.seed = args_cli.seed
    gen = TerrainGenerator(cfg, device="cpu")
    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    gen.terrain_mesh.export(out / "terrain.ply")
    print(f"[메시] {out / 'terrain.ply'}  꼭짓점 {len(gen.terrain_mesh.vertices)}개")

    xs, ys, z = heightmap(gen.terrain_mesh, args_cli.grid)
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        z,
        origin="lower",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        cmap="viridis",
        vmin=np.nanpercentile(z, 1),
        vmax=np.nanpercentile(z, 99),
    )
    fig.colorbar(im, ax=ax, label="height [m]")
    o = gen.terrain_origins.reshape(-1, 3)
    ax.scatter(o[:, 0], o[:, 1], s=4, c="red")
    ax.set_title("construction terrain: rows = difficulty, cols = terrain type (red = spawn origins)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out / "heightmap.png", dpi=120)
    print(f"[높이 지도] {out / 'heightmap.png'}  범위 {np.nanmin(z):.3f} ~ {np.nanmax(z):.3f} m")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
