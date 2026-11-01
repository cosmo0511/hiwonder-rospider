"""Isaac Lab 지형 생성기 설정. **Kit 이 뜬 뒤에 import 해야 한다** (isaaclab.terrains).

course.py 의 numpy 메시를 trimesh 로 감싸 mesh sub-terrain 으로 넣는다. 그레이팅 계단은
구멍이 위아래로 뚫려 있어서 높이맵(height field)으로는 못 그린다 — 그래서 전부 메시 방식이다.

- ``OBSTACLE_TERRAINS_CFG`` : 학습용. 열 = 타일 종류(비율대로), 행 = 난이도.
  기본은 실물 높이 고정이라 행마다 똑같다. ``enable_height_curriculum()`` 을 부르면 행이
  내려갈수록 높이가 커진다 (방지턱/구덩이/계단 한 단에 같은 배율).
- ``REAL_COURSE_TERRAINS_CFG`` : 평가용. 실물 맵 1, 맵 2 두 칸.

[미검증] Isaac Lab 2.3.1 에서 실행해 보지 않았다. SubTerrainBaseCfg 를 상속해 ``function`` 을
지정하는 방식은 isaaclab.terrains.trimesh 의 기본 지형들과 같은 형태로 썼다.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from . import course


def rospider_tile(difficulty: float, cfg: "RoSpiderTileCfg"):
    """sub-terrain 함수. 반환 (trimesh 목록, origin). origin = 로봇 출발점 (타일 좌표)."""
    lo, hi = cfg.height_scale_range
    scale = lo + (hi - lo) * float(np.clip(difficulty, 0.0, 1.0))
    parts, origin = course.build_tile(cfg.kind, cfg.size, scale)
    meshes = [trimesh.Trimesh(vertices=V, faces=F, process=False) for V, F in parts]
    return meshes, origin


@configclass
class RoSpiderTileCfg(SubTerrainBaseCfg):
    function = rospider_tile
    kind: str = "flat"
    """course.SEQUENCES 의 키: flat / bump / pit / pipes / stairs / map1 / map2."""
    height_scale_range: tuple[float, float] = (1.0, 1.0)
    """(난이도 0 일 때 배율, 난이도 1 일 때 배율). 기본 (1, 1) = 항상 실물 높이."""


def _tile(kind: str, proportion: float) -> RoSpiderTileCfg:
    return RoSpiderTileCfg(kind=kind, proportion=proportion)


TRAIN_MIX = {
    # 단독 장애물로 하나씩 익히고, 실물과 같은 순서의 맵에서 이어서 넘는 것도 섞는다.
    "flat": 0.08,
    "bump": 0.14,
    "pit": 0.14,
    "pipes": 0.14,
    "stairs": 0.18,
    "map1": 0.16,
    "map2": 0.16,
}

OBSTACLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=course.TILE,
    border_width=1.0,
    num_rows=6,
    num_cols=14,
    horizontal_scale=0.01,
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    # curriculum=True 여야 열마다 타일 종류가 고정된다 (파이프 배치 이벤트가 이 규칙에 기댄다).
    # 높이를 바꾸지 않는 기본 설정에서도 켜 둔다.
    curriculum=True,
    sub_terrains={k: _tile(k, p) for k, p in TRAIN_MIX.items()},
)

REAL_COURSE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=course.TILE,
    border_width=1.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.01,
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    difficulty_range=(1.0, 1.0),
    sub_terrains={"map1": _tile("map1", 0.5), "map2": _tile("map2", 0.5)},
)


def enable_height_curriculum(gen_cfg: TerrainGeneratorCfg, low: float = 0.5) -> None:
    """행(난이도)에 따라 높이를 low 배 -> 1 배로 키운다. 학습이 안 오를 때만 쓴다."""
    for sub in gen_cfg.sub_terrains.values():
        sub.height_scale_range = (low, 1.0)


def column_kinds(gen_cfg: TerrainGeneratorCfg) -> list[str]:
    """열 번호 -> 타일 종류. 파이프 배치 이벤트가 쓴다."""
    names = list(gen_cfg.sub_terrains.keys())
    props = [gen_cfg.sub_terrains[n].proportion for n in names]
    kinds = [gen_cfg.sub_terrains[n].kind for n in names]
    by_name = dict(zip(names, kinds))
    return [by_name[n] for n in course.column_kinds(names, props, gen_cfg.num_cols)]
