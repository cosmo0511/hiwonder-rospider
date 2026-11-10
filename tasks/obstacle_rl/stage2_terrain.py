"""2단계(방지턱 / 구덩이) 학습 지형. **Kit 이 뜬 뒤에 import.**

평가(obstacle_eval.py)는 원호 방지턱 폭 0.40 ~ 0.60 m, 상자 구덩이 길이 0.25 m 이고, 실물 코스는 사다리꼴 방지턱
0.40 m, 둥근 구덩이 0.50 m 다. 한 모양만 배우면 다른 모양에서 무너질 수 있어서 모양 / 폭을 섞는다.
높이는 행(난이도)에 따라 낮게 시작해 목표보다 조금 높게(방지턱 10 cm, 구덩이 7 cm)까지 올린다.
타일 크기와 로봇 출발점은 course.py 와 같다 (출발점에서 0.70 m 앞이 장애물 중심 = obstacle_eval 과 같음).
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from . import course

BUMP_H_MAX, PIT_D_MAX = 0.10, 0.07
START_X = course.START_LINE + course.ROBOT_FROM_START
OBST_C = START_X + 0.70


def _arc(x, h, w):
    r = (w * w / 4.0 + h * h) / (2.0 * h)
    d = x - w / 2.0
    return np.clip(np.sqrt(np.clip(r * r - d * d, 0.0, None)) - (r - h), 0.0, None)


def stage2_tile(difficulty: float, cfg: "Stage2TileCfg"):
    lo, hi = cfg.scale_range
    k = lo + (hi - lo) * float(np.clip(difficulty, 0.0, 1.0))
    L, W = course.TILE
    parts = []
    origin = np.array([START_X, W / 2, 0.0])
    if cfg.kind == "arc_bump":
        w = np.random.uniform(0.30, 0.70)
        h = max(BUMP_H_MAX * k, 0.005)
        parts.append(course.box(0.0, L, 0.0, W, -course.FLOOR_T, 0.0))
        xs = np.linspace(0.0, w, 61)
        parts.append(course.prism(OBST_C - w / 2 + xs, np.maximum(_arc(xs, h, w), 1e-3), 0.0, 0.0, W))
    elif cfg.kind == "box_pit":
        ln = np.random.uniform(0.20, 0.55)
        d = max(PIT_D_MAX * k, 0.005)
        a, b = OBST_C - ln / 2, OBST_C + ln / 2
        parts.append(course.box(0.0, L, 0.0, W, -course.FLOOR_T, 0.0))
        parts.append(course.box(0.0, a, 0.0, W, 0.0, d))
        parts.append(course.box(b, L, 0.0, W, 0.0, d))
        origin[2] = d
    else:   # course.py 의 실물 모양 (flat / bump 사다리꼴 / pit 둥근 단면). 높이 배율은 실물 기준
        real_k = k * (BUMP_H_MAX / course.BUMP_H if cfg.kind == "bump" else PIT_D_MAX / course.PIT_D)
        parts, origin = course.build_tile(cfg.kind, course.TILE, real_k)
    return [trimesh.Trimesh(vertices=V, faces=F, process=False) for V, F in parts], origin


@configclass
class Stage2TileCfg(SubTerrainBaseCfg):
    function = stage2_tile
    kind: str = "flat"
    """flat / bump / pit (실물 모양) / arc_bump / box_pit (obstacle_sweep 모양, 폭 무작위)"""
    scale_range: tuple[float, float] = (0.2, 1.0)
    """(난이도 0, 난이도 1) 일 때 최대 높이(방지턱 10 cm, 구덩이 7 cm) 에 곱하는 배율."""


STAGE2_MIX = {"flat": 0.10, "arc_bump": 0.25, "bump": 0.20, "box_pit": 0.25, "pit": 0.20}

STAGE2_TERRAINS_CFG = TerrainGeneratorCfg(
    size=course.TILE,
    border_width=1.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.01,
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={k: Stage2TileCfg(kind=k, proportion=p) for k, p in STAGE2_MIX.items()},
)
