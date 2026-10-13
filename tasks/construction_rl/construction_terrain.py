"""공사장 험지 설정 (Isaac Lab 지형 생성기). **Kit 이 뜬 뒤에 import 해야 한다.**

Isaac Lab 기본 험지 `ROUGH_TERRAINS_CFG` 는 ANYmal(큰 4족) 기준이라 계단이 5~23 cm 다.
RosPider 는 몸통 높이 0.12 m, 발 드는 높이 3.5 cm, 보폭 7 cm, 최고 속도 0.12 m/s 라
(`tasks/sensor_fusion/gait.py` 의 GaitParams) 그대로 쓰면 첫 줄부터 못 넘는다.
그래서 지형 종류를 공사장에 있을 법한 것으로 바꾸고, 크기를 로봇에 맞춰 줄였다.

| 이름 | 공사장에서 뭐냐 | Isaac Lab 지형 |
|---|---|---|
| flat | 다져진 바닥 (쉬운 출발점) | MeshPlaneTerrainCfg |
| gravel | 자갈, 흙바닥 | HfRandomUniformTerrainCfg |
| dirt_ramp / dirt_ramp_down | 흙 경사로 | HfPyramidSlopedTerrainCfg / HfInvertedPyramidSlopedTerrainCfg |
| rubble | 잔해, 콘크리트 조각 | HfDiscreteObstaclesTerrainCfg |
| scattered_bricks | 흩어진 벽돌, 자재 | MeshRepeatedBoxesTerrainCfg |
| rebar_stubs | 바닥에 박힌 철근 토막 | MeshRepeatedCylindersTerrainCfg |
| beams | 바닥에 놓인 각재, H빔, 배관 | MeshRailsTerrainCfg |
| curb_stairs / curb_stairs_down | 턱, 비계 계단 | MeshPyramidStairsTerrainCfg / MeshInvertedPyramidStairsTerrainCfg |
| trench | 얕은 구덩이, 배수로 | MeshPitTerrainCfg |
| plate_gap | 철판 사이 틈 | MeshGapTerrainCfg |

[미검증] 아래 숫자는 전부 출발점이다. 확인한 것은 Isaac Lab v2.3.1 소스의 필드 이름과
난이도 보간 방향뿐이다. 이 파일로 지형을 실제로 생성해 본 적은 아직 없다
(`preview_terrain.py` 로 먼저 확인할 것). 특히 높이 상한은 RL 정책이 다리를 얼마나
들 수 있는지에 달렸는데, 그 값은 아직 계산하지 않았다.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# 블록 하나 크기 [m]. ANYmal 예제는 8 m 다. `terrain_levels_vel` 커리큘럼은 에피소드 동안
# 블록 크기의 절반 이상을 가면 다음 단계로 올린다. RosPider 최고 속도 0.12 m/s 로
# 20 s 에피소드면 2.4 m 이므로, 3 m 블록이면 1.5 m 만 가면 승급이다.
TILE_SIZE_M = 3.0

# 가운데 시작 발판 [m]. 로봇 다리 폭이 약 0.4 m 라 그보다 넉넉하게.
PLATFORM_M = 0.6

CONSTRUCTION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(TILE_SIZE_M, TILE_SIZE_M),
    border_width=2.0,
    num_rows=10,  # 난이도 단계 (위로 갈수록 어려움)
    num_cols=24,  # 지형 종류 칸. proportion 비율대로 나눠 가진다
    # 높이맵 지형 해상도. 기본 0.1 m 격자는 발 끝(수 cm)보다 커서 2 cm 로 줄였다.
    horizontal_scale=0.02,
    vertical_scale=0.002,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "gravel": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.004, 0.025),
            noise_step=0.002,
            border_width=0.1,
        ),
        "dirt_ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.25),  # 기울기(tan). 0.25 는 약 14도
            platform_width=PLATFORM_M,
            border_width=0.1,
        ),
        "dirt_ramp_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.25),
            platform_width=PLATFORM_M,
            border_width=0.1,
        ),
        "rubble": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            obstacle_height_mode="choice",  # 튀어나온 것과 파인 것이 섞인다
            obstacle_width_range=(0.06, 0.2),
            obstacle_height_range=(0.01, 0.05),
            num_obstacles=60,
            platform_width=PLATFORM_M,
            border_width=0.1,
        ),
        "scattered_bricks": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.15,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=30, height=0.015, size=(0.12, 0.06), max_yx_angle=0.0
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=90, height=0.05, size=(0.22, 0.10), max_yx_angle=15.0
            ),
            platform_width=PLATFORM_M,
            platform_height=0.0,  # 시작 발판을 바닥 높이에 둔다 (기본 -1 이면 물체 높이를 따라감)
        ),
        "rebar_stubs": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=20, height=0.02, radius=0.01, max_yx_angle=0.0
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=60, height=0.05, radius=0.015, max_yx_angle=10.0
            ),
            platform_width=PLATFORM_M,
            platform_height=0.0,
        ),
        "beams": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1,
            rail_thickness_range=(0.04, 0.10),
            # 주의: rails_terrain 은 `range[1] - 난이도 * (range[1] - range[0])` 로 계산해서
            # (낮, 높) 순서로 넣으면 어려울수록 낮아진다. 어려울수록 높아지게 (높, 낮) 순서로 넣었다.
            rail_height_range=(0.05, 0.01),
            platform_width=PLATFORM_M,
        ),
        "curb_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.01, 0.05),
            step_width=0.25,
            platform_width=PLATFORM_M,
            border_width=0.2,
        ),
        "curb_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.01, 0.05),
            step_width=0.25,
            platform_width=PLATFORM_M,
            border_width=0.2,
        ),
        "trench": terrain_gen.MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.01, 0.05),  # 로봇이 구덩이 바닥에서 시작해 기어 나와야 함
            platform_width=1.2,
        ),
        "plate_gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05,
            gap_width_range=(0.01, 0.05),  # 보폭 7 cm 보다 작게
            platform_width=PLATFORM_M,
        ),
    },
)
"""공사장 험지. 비율 합은 1.0 이다."""


def easy_variant(scale: float = 0.5) -> TerrainGeneratorCfg:
    """높이/폭 상한을 줄인 쉬운 버전. 첫 학습이 안 될 때 이걸로 시작한다.

    `scale` 을 곱하는 대상은 높이, 깊이, 틈 폭, 경사다. 개수와 크기는 그대로 둔다.
    """
    cfg = CONSTRUCTION_TERRAINS_CFG.copy()
    st = cfg.sub_terrains
    st["gravel"].noise_range = (st["gravel"].noise_range[0], st["gravel"].noise_range[1] * scale)
    for k in ("dirt_ramp", "dirt_ramp_down"):
        st[k].slope_range = (0.0, st[k].slope_range[1] * scale)
    st["rubble"].obstacle_height_range = tuple(v * scale for v in st["rubble"].obstacle_height_range)
    st["scattered_bricks"].object_params_end.height *= scale
    st["rebar_stubs"].object_params_end.height *= scale
    st["beams"].rail_height_range = tuple(v * scale for v in st["beams"].rail_height_range)
    for k in ("curb_stairs", "curb_stairs_down"):
        st[k].step_height_range = tuple(v * scale for v in st[k].step_height_range)
    st["trench"].pit_depth_range = tuple(v * scale for v in st["trench"].pit_depth_range)
    st["plate_gap"].gap_width_range = tuple(v * scale for v in st["plate_gap"].gap_width_range)
    return cfg
