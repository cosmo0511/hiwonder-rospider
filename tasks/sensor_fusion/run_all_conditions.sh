#!/usr/bin/env bash
# 과제에 쓸 실험 조건을 순서대로 다 돌린다.
#
# 전제: conda isaac_lab 이 활성화돼 있고(프롬프트에 (isaac_lab)), USD 변환이 끝났다.
#   source /opt/isaaclab-scripts/isaac-env.sh
#   cd /workspace/rospider
#   python tasks/sensor_fusion/02_urdf_to_usd.py
#
# 조건마다 Kit 을 한 번씩 띄우므로 전체 10분 내외다.
# 결과는 outputs/sensor_fusion/ 에 쌓인다.

set -u  # set -e 는 쓰지 않는다. 조건 하나가 실패해도 나머지는 돌려야 한다.

DEMO="tasks/sensor_fusion/run_fusion_demo.py"
EXTRA="${EXTRA:-}"   # 예: EXTRA="--no_rgb --cam_width 96 --cam_height 72" bash ...

if [ ! -f "${DEMO}" ]; then
  echo "[실패] ${DEMO} 가 없다. 레포 루트(/workspace/rospider)에서 실행하자." >&2
  exit 1
fi
if [ ! -f "assets/usd/rospider.usd" ]; then
  echo "[실패] assets/usd/rospider.usd 가 없다. 먼저 변환하자:" >&2
  echo "       python tasks/sensor_fusion/02_urdf_to_usd.py" >&2
  exit 1
fi

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

FAILED=""

# 1) 기준 조건. 수평 자세.
run level             --pitch_deg 0

# 2) 20도 숙임. 장애물과 임계값은 그대로 두고 자세만 바꾼다.
#    융합은 거리를 유지하고, 고정 ROI 기준선은 무너져야 한다.
run pitch20           --pitch_deg 20

# 3) 넘어갈 수 있는 2 cm 단차. 융합은 장애물로 세지 않아야 한다.
run lowstep           --pitch_deg 0  --obstacle_height 0.02

# 4) 바닥 추정을 끄고 설치 높이 상수를 믿게 한다. 의도된 실패다 —
#    기울면 높이가 밀려 바닥이 장애물로 올라온다.
run pitch20_assumed   --pitch_deg 20 --floor_mode assumed_height

echo
echo "=============================================================="
if [ -n "${FAILED}" ]; then
  echo "실패한 조건:${FAILED}"
else
  echo "조건 4개 전부 완료"
fi
echo "결과: outputs/sensor_fusion/"
ls -1 outputs/sensor_fusion/ 2>/dev/null | head -30
