#!/usr/bin/env python3
"""course.py 검사. **numpy 만 쓴다 — Kit 없이 아무 파이썬에서 돈다.**

    python tasks/obstacle_rl/test_course.py

A. 메시 조각이 전부 닫힌 솔리드이고 부피가 양수인가 (법선 방향이 맞는가)
B. 맵 1/2 배치가 최종 설계 수치와 같은가 (출발선 ~ 작업대 끝 2.7972 / 2.6972 m)
C. 윗면 높이가 실물 수치와 같은가 (방지턱 0.08, 구덩이 -0.047, 계단 0.06 x 4, 작업대 0.24)
D. 그레이팅 구멍이 실물(12.7 x 69.3 mm)과 거의 같은가
E. 타일이 타일 크기 안에 들어가는가
F. 높이 배율(scale)이 방지턱/구덩이/계단에만 걸리는가
G. 파이프 시작 위치와 지형 생성기 열 -> 타일 종류 규칙
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import course as C  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


def closed_and_positive(V, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fw = Counter(map(tuple, e))
    open_edges = sum(1 for (a, b), k in fw.items() if fw.get((b, a), 0) != k)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    return open_edges, vol


# A
for kind in C.SEQUENCES:
    parts, _ = C.build_tile(kind)
    bad = [(i, *closed_and_positive(V, F)) for i, (V, F) in enumerate(parts)]
    bad = [b for b in bad if b[1] != 0 or b[2] <= 0]
    check(f"A {kind}: 조각 {len(parts)}개 모두 닫힘 + 부피 양수", not bad, f"문제 조각 {bad[:3]}" if bad else "")

# B
for kind, want in (("map1", 2.7972), ("map2", 2.6972)):
    lay = C.layout(kind)
    got = lay["plat"] + C.PLAT_L - C.START_LINE
    check(f"B {kind}: 출발선 ~ 작업대 끝 {got:.4f} m", abs(got - want) < 1e-6, f"설계 {want}")
lay = C.layout("map1")
check("B map1 순서 구덩이 -> 파이프 -> 계단", lay["pit"] < lay["pipes"] < lay["stairs"])
check("B 출발선 ~ 첫 장애물 0.60", abs(lay["pit"] - C.START_LINE - 0.60) < 1e-9)
check("B 파이프 앞뒤 간격 0.15", abs(lay["pipes"] - (lay["pit"] + C.PIT_L) - 0.15) < 1e-9
      and abs(lay["stairs"] - (lay["pipes"] + C.PIPE_SPAN) - 0.15) < 1e-9)

# C
lay2 = C.layout("map2")
z = lambda k, x: float(C.surface_z(k, x))  # noqa: E731
check("C 방지턱 꼭대기 0.08", abs(z("map2", lay2["bump"] + 0.2) - 0.08) < 1e-9)
check("C 구덩이 바닥 -0.047", abs(z("map1", lay["pit"] + 0.25) + 0.047) < 1e-9)
check("C 구덩이 테두리 0", abs(z("map1", lay["pit"] + 0.003)) < 1e-9)
steps = [z("map1", lay["stairs"] + C.STAIR_TREAD * (k + 0.5)) for k in range(4)]
check("C 계단 0.06/0.12/0.18/0.24", np.allclose(steps, [0.06, 0.12, 0.18, 0.24]), str(steps))
check("C 작업대 0.24", abs(z("map1", lay["plat"] + 0.3) - 0.24) < 1e-9)
check("C 작업대 뒤 0 (책상 끝처럼 떨어짐)", abs(z("map1", lay["plat"] + C.PLAT_L + 0.05)) < 1e-9)

# D
_, ox, oy = C._grating(0.0, 0.06, C.TILE[1])
check("D 그레이팅 구멍 진행 폭 12.7 mm", abs(ox * 1000 - 12.67) < 0.1, f"{ox * 1000:.2f} mm")
check("D 그레이팅 구멍 가로 길이 69 mm 근처", abs(oy * 1000 - 69.3) < 3.0, f"{oy * 1000:.1f} mm")
check("D 발끝(지름 18) 이 구멍에 들어가는 깊이 < 3 mm",
      9.0 - np.sqrt(81 - (ox * 500) ** 2) < 3.0, f"{9.0 - np.sqrt(81 - (ox * 500) ** 2):.2f} mm")

# E
for kind in C.SEQUENCES:
    V, _ = C.merge(C.build_tile(kind)[0])
    inside = V[:, 0].min() >= -1e-9 and V[:, 0].max() <= C.TILE[0] + 1e-9 and V[:, 1].min() >= -1e-9 \
        and V[:, 1].max() <= C.TILE[1] + 1e-9
    check(f"E {kind}: 타일 안, z {V[:, 2].min():.3f} ~ {V[:, 2].max():.3f}", inside)

# F
check("F 배율 0.5: 방지턱 0.04", abs(float(C.surface_z("map2", lay2["bump"] + 0.2, 0.5)) - 0.04) < 1e-9)
check("F 배율 0.5: 작업대 0.12", abs(float(C.surface_z("map1", lay["plat"] + 0.3, 0.5)) - 0.12) < 1e-9)
V, _ = C.merge(C.build_tile("map1", scale=0.5)[0])
check("F 배율 0.5: 세운 판 꼭대기 0.12 + 0.15", abs(V[:, 2].max() - 0.27) < 1e-9, f"{V[:, 2].max():.3f}")

# G
pc = C.pipe_centers("map1")
o = C.origin("map1")[0]
check("G 맵 1 첫 파이프 중심 = 파이프 시작 + 반지름",
      abs(pc[0] + o - (lay["pipes"] + C.PIPE_D / 2)) < 1e-9, f"출발점 기준 {pc[0]:.4f} m")
check("G 파이프 사이 간격 0.10", abs(pc[1] - pc[0] - (C.PIPE_D + 0.10)) < 1e-9)
check("G 파이프 없는 타일은 None", C.pipe_centers("bump") is None and C.pipe_centers("stairs") is None)
check("G 파이프 질량 약 2.08 kg", abs(C.PIPE_MASS - 2.08) < 0.02, f"{C.PIPE_MASS:.3f} kg")
names = ["a", "b", "c"]
cols = C.column_kinds(names, [0.2, 0.3, 0.5], 10)
check("G 열 규칙: 비율대로 열이 나뉨", Counter(cols) == Counter({"a": 2, "b": 3, "c": 5}), str(cols))

print()
print("전부 통과" if not FAIL else f"실패 {len(FAIL)}개: {FAIL}")
sys.exit(1 if FAIL else 0)
