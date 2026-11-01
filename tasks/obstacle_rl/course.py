"""장애물 코스 지형 기하. **numpy 만 쓴다 (Kit / Isaac Lab 불필요).**

실물 코스(2026-11 최종 설계, rospider-isaacsim 레포 cad/ 와 3D 도면 페이지 기준)를
강화학습 타일로 옮긴다. 메시는 (V, F) numpy 배열 목록으로 만들고, Isaac Lab 쪽
(`terrain_cfg.py`)에서 trimesh 로 감싸 지형 생성기에 넣는다. 그래서 이 파일은
컨테이너 밖에서도 `test_course.py` 로 검사할 수 있다.

실물 치수 (단위 m, 높이는 바닥 윗면 = 0 기준)
------------------------------------------------
| 장애물 | 진행 길이 | 높이 | 비고 |
|---|---|---|---|
| 과속방지턱 | 0.40 | 0.08 | 사다리꼴, 윗면 0.12, 가장자리 3 mm |
| 구덩이 | 0.50 | -0.047 | 둥근 단면: 테두리 5 mm + 곡면 150 mm + 평평 190 mm |
| 파이프 2개 | 0.1972 | 0.0486 | 지름 48.6, 길이 1.0, 간격 0.10. **굴러간다** -> 지형이 아니라 물체 |
| 그레이팅 계단 | 0.60 | 0.06 x 4 = 0.24 | 위 30 mm 그레이팅(구멍 12.7 x 69.3, 위아래 관통), 아래 MDF |
| 작업대 | 0.60 | 0.24 | 계단 뒤. 끝에 ㄴ자 용접판 (가정값 0.20 x 0.60, 세운 판 0.15) |

맵 1 = 구덩이 -> 파이프 -> 계단 -> 작업대, 맵 2 = 방지턱 -> 파이프 -> 계단 -> 작업대.
출발선 ~ 첫 장애물 0.60, 파이프 앞뒤 간격 0.15.

타일 좌표
---------
타일은 x(진행) 0 ~ TILE[0], y(가로) 0 ~ TILE[1], 바닥 윗면 z = 0.
출발선은 x = START_LINE, 로봇 출발점(지형 생성기의 origin)은 출발선 + ROBOT_FROM_START.
장애물은 가로 전체(타일 폭)에 걸친다 — 옆으로 돌아가지 못하게. 파이프만 1.0 m 라
양옆이 0.1 m 씩 빈다(실물은 코스 1.1 m 에 0.05 m 씩).

높이 고정 vs 커리큘럼
---------------------
기본은 **실물 높이 고정**(scale = 1). `build_tile(..., scale=s)` 로 방지턱 높이,
구덩이 깊이, 계단 한 단 높이에 같은 배율을 곱할 수 있다. 학습이 안 오를 때
terrain_cfg.py 의 height_scale_range 를 (0.5, 1.0) 처럼 주면 난이도(행)에 따라
낮은 것부터 시작한다. 그레이팅 두께(30 mm)와 판 크기는 배율과 무관하다.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- 실물 치수 (m)
BUMP_L, BUMP_H, BUMP_TOP, BUMP_EDGE = 0.40, 0.08, 0.12, 0.003
PIT_L, PIT_D, PIT_LIP, PIT_RAMP = 0.50, 0.047, 0.005, 0.15
PIPE_D, PIPE_LEN, PIPE_GAP, PIPE_WALL = 0.0486, 1.0, 0.10, 0.0018
PIPE_SPAN = 2 * PIPE_D + PIPE_GAP
PIPE_MASS = math.pi * (PIPE_D - PIPE_WALL) * PIPE_WALL * PIPE_LEN * 7850.0
"""강관 Ø48.6 x 1.8T x 1 m 의 질량 [kg] (밀도 7850 으로 계산, 약 2.08). 실측 전 계산값."""

STAIR_N, STAIR_TREAD, STAIR_RISE, GRATE_T = 4, 0.15, 0.06, 0.03
GRATE_NOSE, GRATE_BACK, GRATE_BAR_X, GRATE_BAR_Y = 0.008, 0.004, 0.003, 0.004
GRATE_HOLES_X = 9
GRATE_HOLE_Y = 0.0693
PLAT_L = 0.60
PLATE_L, PLATE_W, PLATE_H, PLATE_T, PLATE_MARGIN = 0.20, 0.60, 0.15, 0.003, 0.02

START_ZONE = 0.60   # 출발선 ~ 첫 장애물
PIPE_GAP_AROUND = 0.15   # 파이프 앞뒤 간격 (맵 1, 2 모두 파이프가 가운데)

# ---------------------------------------------------------------- 타일
TILE = (4.0, 1.2)
"""타일 크기 (진행, 가로) [m]. 맵 1(출발선부터 2.80 m)이 들어가고 가로는 코스 폭(1.1)에 가깝게."""
START_LINE = 0.20
ROBOT_FROM_START = 0.30
"""출발선에서 로봇 중심까지. 몸 길이가 약 0.4 m 라 뒷다리가 출발선 근처에 온다."""
FLOOR_T = 0.10
"""바닥 판 두께 [m] (z = -FLOOR_T ~ 0)."""

SEQUENCES = {
    "flat": [],
    "bump": ["bump"],
    "pit": ["pit"],
    "pipes": ["pipes"],
    "stairs": ["stairs"],
    "map1": ["pit", "pipes", "stairs"],
    "map2": ["bump", "pipes", "stairs"],
}
"""타일 종류 -> 장애물 순서. 계단이 있으면 그 뒤에 작업대(+ ㄴ자 판)가 붙는다."""

LENGTH = {"bump": BUMP_L, "pit": PIT_L, "pipes": PIPE_SPAN, "stairs": STAIR_N * STAIR_TREAD}


def layout(kind: str) -> dict[str, float]:
    """타일 x 좌표로 장애물 시작 위치. 계단이 있으면 'plat' 도 넣는다."""
    seq = SEQUENCES[kind]
    x = START_LINE + START_ZONE
    out: dict[str, float] = {}
    for i, k in enumerate(seq):
        out[k] = x
        x += LENGTH[k]
        if i < len(seq) - 1:
            x += PIPE_GAP_AROUND
    if "stairs" in out:
        out["plat"] = x
    return out


def origin(kind: str) -> np.ndarray:
    """로봇 출발점 (타일 좌표). 지형 생성기가 이 점을 env origin 으로 쓴다."""
    return np.array([START_LINE + ROBOT_FROM_START, TILE[1] / 2, 0.0])


def pipe_centers(kind: str) -> list[float] | None:
    """파이프 두 개의 시작 중심 x (로봇 출발점 기준 상대값). 파이프가 없는 타일은 None."""
    lay = layout(kind)
    if "pipes" not in lay:
        return None
    o = origin(kind)[0]
    r = PIPE_D / 2
    return [lay["pipes"] + r - o, lay["pipes"] + r + PIPE_D + PIPE_GAP - o]


PIPE_PARK = [-0.40, -0.47]
"""파이프가 없는 타일에서 파이프를 치워 두는 곳 (출발점 기준 x, 로봇 뒤). 전진 명령만 주므로 닿을 일이 적다."""


# ---------------------------------------------------------------- 단면
def bump_z(s, h=BUMP_H):
    s = np.asarray(s, dtype=float)
    ramp = (BUMP_L - BUMP_TOP) / 2
    d = np.minimum(s, BUMP_L - s)
    z = np.where(d >= ramp, h, BUMP_EDGE + (h - BUMP_EDGE) * np.clip(d, 0, None) / ramp)
    return np.where((s < 0) | (s > BUMP_L), 0.0, z)


def pit_z(s, depth=PIT_D):
    s = np.asarray(s, dtype=float)
    d = np.minimum(s, PIT_L - s)
    t = np.clip((d - PIT_LIP) / PIT_RAMP, 0.0, 1.0)
    z = -depth * np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))
    z = np.where(d <= PIT_LIP, 0.0, z)
    return np.where((s < 0) | (s > PIT_L), 0.0, z)


def surface_z(kind: str, x, scale: float = 1.0):
    """타일 윗면 높이 (그레이팅 구멍과 파이프, ㄴ자 판은 빼고). 테스트와 미리보기용."""
    x = np.asarray(x, dtype=float)
    lay = layout(kind)
    z = np.zeros_like(x)
    if "bump" in lay:
        z = z + bump_z(x - lay["bump"], BUMP_H * scale)
    if "pit" in lay:
        z = z + pit_z(x - lay["pit"], PIT_D * scale)
    if "stairs" in lay:
        rise = STAIR_RISE * scale
        s = x - lay["stairs"]
        k = np.clip(np.floor(s / STAIR_TREAD) + 1, 0, STAIR_N)
        z = np.where((s >= 0) & (s < STAIR_N * STAIR_TREAD), k * rise, z)
        z = np.where((x >= lay["plat"]) & (x <= lay["plat"] + PLAT_L), STAIR_N * rise, z)
    return z


# ---------------------------------------------------------------- 메시 조각
def box(x0, x1, y0, y1, z0, z1):
    """축 정렬 상자 (V, F). 바깥 법선."""
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def prism(xs, ztop, zbot, y0, y1):
    """x 방향으로 단조인 단면(윗면 ztop(x), 바닥 zbot)을 y0~y1 로 늘린 닫힌 솔리드."""
    xs = np.asarray(xs, dtype=float)
    zt = np.asarray(ztop, dtype=float)
    n = len(xs)
    # 꼭짓점: [top y0, top y1, bot y0, bot y1] x n
    tA = np.stack([xs, np.full(n, y0), zt], 1)
    tB = np.stack([xs, np.full(n, y1), zt], 1)
    bA = np.stack([xs, np.full(n, y0), np.full(n, zbot)], 1)
    bB = np.stack([xs, np.full(n, y1), np.full(n, zbot)], 1)
    V = np.concatenate([tA, tB, bA, bB])
    TA, TB, BA, BB = 0, n, 2 * n, 3 * n
    i = np.arange(n - 1)
    j = i + 1
    F = [
        np.stack([TA + i, TA + j, TB + j], 1), np.stack([TA + i, TB + j, TB + i], 1),   # 윗면
        np.stack([BA + i, BB + j, BA + j], 1), np.stack([BA + i, BB + i, BB + j], 1),   # 바닥
        np.stack([BA + i, BA + j, TA + j], 1), np.stack([BA + i, TA + j, TA + i], 1),   # y0 옆면 (-y)
        np.stack([BB + i, TB + j, BB + j], 1), np.stack([BB + i, TB + i, TB + j], 1),   # y1 옆면 (+y)
        np.array([[BA, TB, BB], [BA, TA, TB]]),                                         # x 시작 (-x)
        np.array([[BA + n - 1, BB + n - 1, TB + n - 1], [BA + n - 1, TB + n - 1, TA + n - 1]]),  # x 끝 (+x)
    ]
    return V, np.concatenate(F)


def _floor(x0, x1, W):
    return box(x0, x1, 0.0, W, -FLOOR_T, 0.0)


def _bump(x0, W, h):
    ramp = (BUMP_L - BUMP_TOP) / 2
    xs = [x0, x0 + ramp, x0 + ramp + BUMP_TOP, x0 + BUMP_L]
    zt = [BUMP_EDGE * min(1.0, h / BUMP_H), h, h, BUMP_EDGE * min(1.0, h / BUMP_H)]
    return prism(xs, zt, 0.0, 0.0, W)


def _pit(x0, W, depth, n=80):
    s = np.unique(np.concatenate([np.linspace(0, PIT_L, n + 1),
                                  [PIT_LIP, PIT_LIP + PIT_RAMP, PIT_L - PIT_LIP - PIT_RAMP, PIT_L - PIT_LIP]]))
    return prism(x0 + s, pit_z(s, depth), -FLOOR_T, 0.0, W)


def _grating(x0, z_top, W):
    """그레이팅 한 단 (위아래 관통, 두께 GRATE_T). 상자 여러 개."""
    parts = []
    z0 = z_top - GRATE_T
    span = STAIR_TREAD - GRATE_NOSE - GRATE_BACK
    ox = (span - (GRATE_HOLES_X - 1) * GRATE_BAR_X) / GRATE_HOLES_X
    parts.append(box(x0, x0 + GRATE_NOSE, 0.0, W, z0, z_top))
    parts.append(box(x0 + STAIR_TREAD - GRATE_BACK, x0 + STAIR_TREAD, 0.0, W, z0, z_top))
    for i in range(1, GRATE_HOLES_X):   # 진행 방향 사이 살 (가로로 길게)
        xb = x0 + GRATE_NOSE + i * ox + (i - 1) * GRATE_BAR_X
        parts.append(box(xb, xb + GRATE_BAR_X, 0.0, W, z0, z_top))
    n_y = max(1, round((W - GRATE_BAR_Y) / (GRATE_HOLE_Y + GRATE_BAR_Y)))
    oy = (W - (n_y + 1) * GRATE_BAR_Y) / n_y
    xa, xb = x0 + GRATE_NOSE, x0 + STAIR_TREAD - GRATE_BACK
    for j in range(n_y + 1):            # 가로 방향 사이 살 (진행 방향으로 짧게)
        y = j * (oy + GRATE_BAR_Y)
        parts.append(box(xa, xb, y, y + GRATE_BAR_Y, z0, z_top))
    return parts, ox, oy


def _stairs_and_platform(x0, W, rise, with_plate=True):
    parts = []
    for k in range(STAIR_N):
        xs = x0 + k * STAIR_TREAD
        top = (k + 1) * rise
        base = top - GRATE_T
        if base > 0:
            parts.append(box(xs, xs + STAIR_TREAD, 0.0, W, 0.0, base))      # MDF 받침
        g, _, _ = _grating(xs, top, W)
        parts.extend(g)
    xp = x0 + STAIR_N * STAIR_TREAD
    H = STAIR_N * rise
    parts.append(box(xp, xp + PLAT_L, 0.0, W, 0.0, H))                       # 작업대
    if with_plate:
        px = xp + PLAT_L - PLATE_MARGIN - PLATE_L
        y0, y1 = (W - PLATE_W) / 2, (W + PLATE_W) / 2
        parts.append(box(px, px + PLATE_L, y0, y1, H, H + PLATE_T))          # ㄴ자 바닥 판
        parts.append(box(px + PLATE_L - PLATE_T, px + PLATE_L, y0, y1, H, H + PLATE_H))   # 세운 판
    return parts


def build_tile(kind: str, size=TILE, scale: float = 1.0):
    """타일 하나. 반환 (parts, origin): parts = [(V, F), ...] 타일 좌표 [m], origin = 로봇 출발점.

    scale 은 방지턱 높이 / 구덩이 깊이 / 계단 한 단 높이에 곱한다 (기본 1 = 실물).
    """
    if tuple(size) != tuple(TILE):
        raise ValueError(f"타일 크기는 {TILE} 로 고정이다 (받은 값 {size}). 배치가 이 크기에 맞춰져 있다.")
    L, W = TILE
    lay = layout(kind)
    parts = []
    # 바닥: 구덩이 자리만 비움
    if "pit" in lay:
        parts.append(_floor(0.0, lay["pit"], W))
        parts.append(_pit(lay["pit"], W, PIT_D * scale))
        parts.append(_floor(lay["pit"] + PIT_L, L, W))
    else:
        parts.append(_floor(0.0, L, W))
    if "bump" in lay:
        parts.append(_bump(lay["bump"], W, BUMP_H * scale))
    if "stairs" in lay:
        parts.extend(_stairs_and_platform(lay["stairs"], W, STAIR_RISE * scale))
    # 출발선 표시 (아주 얇은 띠, 1 mm)
    parts.append(box(START_LINE - 0.01, START_LINE + 0.01, 0.05, W - 0.05, 0.0, 0.001))
    return parts, origin(kind)


def merge(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


# ---------------------------------------------------------------- 지형 생성기 열 -> 타일 종류
def column_kinds(names: list[str], proportions: list[float], num_cols: int) -> list[str]:
    """커리큘럼 모드 지형 생성기가 열마다 어느 sub-terrain 을 까는지.

    Isaac Lab ``TerrainGenerator._generate_curriculum_terrains`` 와 같은 규칙으로 계산한다
    (v2.3.1 소스를 기억해 옮긴 것. 이 컨테이너에서는 대조하지 못했다):
        sub_index = min(where(col / num_cols + 0.001 < cumsum(proportions)))
    """
    p = np.asarray(proportions, dtype=float)
    p = np.cumsum(p / p.sum())
    return [names[int(np.min(np.where(c / num_cols + 0.001 < p)[0]))] for c in range(num_cols)]
