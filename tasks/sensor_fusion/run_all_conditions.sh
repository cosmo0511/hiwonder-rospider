#!/usr/bin/env bash
# 과제에 쓸 실험 조건을 순서대로 다 돌린다.
#
# 전제: conda isaac_lab 이 활성화돼 있고(프롬프트에 (isaac_lab)), USD 변환이 끝났다.
#   source /opt/isaaclab-scripts/isaac-env.sh
#   cd /workspace/rospider
#   python tasks/sensor_fusion/02_urdf_to_usd.py
#
# 조건마다 Kit 을 한 번씩 띄우므로 전체 15분 내외다.
# 결과는 outputs/sensor_fusion/ 에 쌓이고, 마지막에 summary.csv 로 묶인다.
#
# **조건마다 --tag 가 다르다.** 그래야 로그가 {tag}_log.csv 로 따로 남는다.
# 직접 돌릴 때 --tag 를 안 바꾸면 앞 조건의 CSV 가 덮어써진다 — 실제로 한 번 날렸다.

set -u  # set -e 는 쓰지 않는다. 조건 하나가 실패해도 나머지는 돌려야 한다.

DEMO="tasks/sensor_fusion/run_fusion_demo.py"
OUT="outputs/sensor_fusion"
EXTRA="${EXTRA:-}"   # 예: EXTRA="--no_rgb --cam_width 96 --cam_height 72" bash ...
WALK_PHYSICS="${WALK_PHYSICS:-0}"   # 1 이면 물리 보행 조건도 돌린다(넘어질 수 있다)

if [ ! -f "${DEMO}" ]; then
  echo "[실패] ${DEMO} 가 없다. 레포 루트(/workspace/rospider)에서 실행하자." >&2
  exit 1
fi
if [ ! -f "assets/usd/rospider.usd" ]; then
  echo "[실패] assets/usd/rospider.usd 가 없다. 먼저 변환하자:" >&2
  echo "       python tasks/sensor_fusion/02_urdf_to_usd.py" >&2
  exit 1
fi

FAILED=""

run() {
  local tag="$1"; shift
  echo
  echo "=============================================================="
  echo "조건 ${tag}  :  $*"
  echo "=============================================================="
  python "${DEMO}" --enable_cameras --tag "${tag}" ${EXTRA} "$@"
  if [ $? -ne 0 ]; then
    echo "[경고] 조건 ${tag} 가 실패했다. 다음 조건으로 넘어간다." >&2
    FAILED="${FAILED} ${tag}"
  fi
}

# ---------------------------------------------------------------- 보행 조건
# 과제의 시나리오 그대로다. 로봇이 걸어가 장애물 앞에서 스스로 멈춘다.
# 몸통 끄덕임이 보행에서 나오므로 --pitch_deg 로 손수 기울이지 않는다.

# 1) 걸어서 접근 -> DANGER 에서 정지. 이게 대표 결과다.
run walk              --walk --auto_walk 0.10,0,0 --walk_seconds 16 --record

# 2) 같은 보행인데 **정지를 끈다.** 판정만 기록하고 장애물을 통과한다.
#    1번과 나란히 두면 "판정이 행동으로 이어졌다" 가 보인다.
run walk_nostop       --walk --auto_walk 0.10,0,0 --walk_seconds 16 --no_stop_on_danger

# 3) 넘어갈 수 있는 2 cm 단차를 향해 걷는다. 멈추지 **않아야** 한다.
run walk_lowstep      --walk --auto_walk 0.10,0,0 --walk_seconds 16 --obstacle_height 0.02

# 4) 바닥 추정을 끄고 설치 높이 상수를 믿게 한 채로 걷는다. 의도된 실패 —
#    걸으면 몸이 끄덕여 카메라 높이가 변하는데 상수를 믿으니 바닥이 올라온다.
run walk_assumed      --walk --auto_walk 0.10,0,0 --walk_seconds 16 \
                      --floor_mode assumed_height

# ---------------------------------------------------------------- 정지 조건
# 숫자를 깨끗하게 뽑는 쪽. 보행은 매 프레임 자세가 달라 표로 만들기 번거롭다.

# 5) 기준 조건. 수평 자세, 거리 스윕.
run level             --pitch_deg 0

# 6) 20도 숙임. 장애물과 임계값은 그대로 두고 자세만 바꾼다.
#    융합은 거리를 유지하고, 고정 ROI 기준선은 무너져야 한다.
run pitch20           --pitch_deg 20

# 7) 넘어갈 수 있는 2 cm 단차. 융합은 장애물로 세지 않아야 한다.
run lowstep           --pitch_deg 0  --obstacle_height 0.02

# 8) 바닥 추정을 끈 정지 버전.
run pitch20_assumed   --pitch_deg 20 --floor_mode assumed_height

# 9) 계단. 높이맵이 단별로 갈라지는 그림이 나온다.
run stairs            --scene stairs --pitch_deg 0

# ---------------------------------------------------------------- 물리 보행 (선택)
if [ "${WALK_PHYSICS}" = "1" ]; then
  if [ ! -f "assets/usd/rospider_float.usd" ]; then
    echo "[건너뜀] 물리 보행에는 베이스 고정 안 한 USD 가 필요하다:" >&2
    echo "         python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base" >&2
  else
    # PD 게인과 마찰이 [미검증]이라 넘어질 수 있다. 넘어지면 그것도 결과다 —
    # IMU 기울기가 25도를 넘어 STOP_TILT 가 걸리는 게 보인다.
    run walk_physics  --walk --walk_mode physics --usd assets/usd/rospider_float.usd \
                      --auto_walk 0.08,0,0 --walk_seconds 16 --record
  fi
else
  echo
  echo "[안내] 물리 보행 조건은 기본으로 건너뛴다. 돌리려면:"
  echo "       python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base"
  echo "       WALK_PHYSICS=1 bash tasks/sensor_fusion/run_all_conditions.sh"
fi

# ---------------------------------------------------------------- 요약 묶기
# 조건별 CSV 는 열 구성이 서로 다르다(보행 쪽에만 cmd/phase/d_true 가 있다).
# 그래서 합집합 헤더로 묶고, 없는 칸은 비워 둔다. 노션 표에 그대로 붙일 수 있다.
echo
echo "=============================================================="
python - "${OUT}" <<'PY'
import csv, sys
from pathlib import Path

out = Path(sys.argv[1])
logs = sorted(p for p in out.glob("*_log.csv") if p.name != "summary.csv")
if not logs:
    print(f"[요약] {out} 에 *_log.csv 가 없다. 조건이 하나도 안 돌았다.")
    raise SystemExit(0)

rows, fields = [], []
for path in logs:
    tag = path.name[: -len("_log.csv")]
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["tag"] = tag
            rows.append(row)
            for key in row:
                if key not in fields:
                    fields.append(key)

fields = ["tag"] + [f for f in fields if f != "tag"]
summary = out / "summary.csv"
with summary.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields, restval="")
    writer.writeheader()
    writer.writerows(rows)
print(f"[요약] {summary}  ({len(logs)} 조건, {len(rows)} 행)")

# 조건마다 마지막 줄만 뽑아 한눈에 본다.
print()
print(f"{'조건':16s}{'tilt':>7s}{'d_fused':>9s}{'d_naive':>9s}{'d_roi':>8s}"
      f"{'참값':>8s}  {'융합':<12s}{'뎁스단독':<9s}{'고정ROI'}")
last: dict[str, dict] = {}
for row in rows:
    last[row["tag"]] = row
for tag, row in last.items():
    def g(key, default="-"):
        v = row.get(key, "")
        return v if v not in ("", None) else default
    print(f"{tag:16s}{g('tilt_deg'):>7s}{g('d_fused_m'):>9s}{g('d_naive_m'):>9s}"
          f"{g('d_roi_m'):>8s}{g('d_true_m'):>8s}  {g('decision_fused'):<12s}"
          f"{g('decision_naive'):<9s}{g('decision_roi')}")
PY

echo
if [ -n "${FAILED}" ]; then
  echo "실패한 조건:${FAILED}"
else
  echo "조건 전부 완료"
fi
echo "결과: ${OUT}/"
ls -1 "${OUT}/" 2>/dev/null | head -40
