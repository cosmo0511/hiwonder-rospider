#!/usr/bin/env python3
"""타일 미리보기 그림. **numpy + matplotlib 만 쓴다 (Kit 불필요).**

    python tasks/obstacle_rl/preview_tiles.py            # docs/images/obstacle_rl_tiles.png

타일 7종의 진행 방향 단면(실제 비율)과 맵 1 위에서 본 그림(그레이팅 구멍 포함)을 그린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import course as C  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main(out: Path = REPO / "docs" / "images" / "obstacle_rl_tiles.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = list(C.SEQUENCES)
    fig, axs = plt.subplots(len(kinds) + 1, 1, figsize=(13, 2.0 * len(kinds) + 3),
                            gridspec_kw={"height_ratios": [1] * len(kinds) + [2.2]})
    x = np.linspace(0, C.TILE[0], 4001)
    for ax, k in zip(axs, kinds):
        z = C.surface_z(k, x)
        ax.fill_between(x, -0.06, z, color="#c9a46e", lw=0)
        ax.plot(x, z, "k", lw=0.8)
        pc = C.pipe_centers(k)
        o = C.origin(k)
        if pc is not None:
            for c in pc:
                ax.add_patch(plt.Circle((o[0] + c, C.PIPE_D / 2), C.PIPE_D / 2, fc="#9aa3aa", ec="k", lw=0.6))
        lay = C.layout(k)
        if "plat" in lay:
            px = lay["plat"] + C.PLAT_L - C.PLATE_MARGIN - C.PLATE_L
            H = C.STAIR_N * C.STAIR_RISE
            ax.add_patch(plt.Rectangle((px, H), C.PLATE_L, C.PLATE_T, fc="#a8473b"))
            ax.add_patch(plt.Rectangle((px + C.PLATE_L - C.PLATE_T, H), C.PLATE_T, C.PLATE_H, fc="#a8473b"))
        ax.axvline(C.START_LINE, color="#d9682b", lw=1)
        ax.plot([o[0]], [0.14], "v", color="#1d2a33", ms=7)
        ax.set_xlim(0, C.TILE[0])
        ax.set_ylim(-0.06, 0.42)
        ax.set_aspect("equal")
        ax.set_yticks([0, 0.24])
        ax.text(0.02, 0.33, k, fontsize=11, weight="bold")
        ax.grid(alpha=0.25)
    axs[0].set_title("tiles, side section to scale (m). orange = start line, triangle = robot spawn (env origin)")
    # 맵 1 위에서
    ax = axs[-1]
    parts, o = C.build_tile("map1")
    for V, F in parts:
        x0, x1 = V[:, 0].min(), V[:, 0].max()
        y0, y1 = V[:, 1].min(), V[:, 1].max()
        ztop = V[:, 2].max()
        if ztop <= 0.0011 and x1 - x0 > 0.5:
            c = "#e6dccb"
        elif ztop <= 0.0011:
            c = "#d9682b"
        elif ztop > 0.2 and abs((y1 - y0) - C.PLATE_W) < 1e-6:
            c = "#a8473b"
        else:
            c = plt.cm.YlGn(0.3 + ztop * 2.5)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=c, ec="none"))
    lay = C.layout("map1")
    ax.add_patch(plt.Rectangle((lay["pit"], 0), C.PIT_L, C.TILE[1], fc="#b9bec2", ec="none"))
    for c in C.pipe_centers("map1"):
        ax.add_patch(plt.Rectangle((o[0] + c - C.PIPE_D / 2, (C.TILE[1] - C.PIPE_LEN) / 2), C.PIPE_D, C.PIPE_LEN,
                                   fc="#9aa3aa", ec="k", lw=0.5))
    ax.plot([o[0]], [o[1]], "o", color="#1d2a33")
    ax.set_xlim(0, C.TILE[0])
    ax.set_ylim(0, C.TILE[1])
    ax.set_aspect("equal")
    ax.set_title("map1 from above (grating holes 12.7 x 70.8 mm visible as stripes)")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
