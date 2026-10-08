#!/usr/bin/env bash
# 설치가 제대로 됐는지 컨테이너 안에서 확인합니다.
#   docker compose run --rm isaaclab bash /workspace/rospider/docker/isaaclab/smoke_test.sh
set -uo pipefail

fail=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== 1. 버전 핀 =="
source /opt/isaaclab-scripts/isaac-env.sh
python - <<'PY'
import sys, torch
print(f"  python       {sys.version.split()[0]}")
print(f"  torch        {torch.__version__}  (cuda {torch.version.cuda})")
print(f"  cuda 사용가능 {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          {torch.cuda.get_device_name(0)}")
PY
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" \
  && ok "python 3.11" || bad "python 3.11 아님"
python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.7.0') and torch.version.cuda=='12.8' else 1)" \
  && ok "torch 2.7.0 + cu128" || bad "torch 핀 어긋남"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  && ok "GPU 접근" || bad "GPU 안 보임 (--gpus all / nvidia-container-toolkit 확인)"

echo
echo "== 2. Isaac Sim / Isaac Lab import =="
python -c "import isaacsim" 2>/dev/null && ok "isaacsim import" || bad "isaacsim import 실패"
python -c "import isaaclab, isaaclab_tasks" 2>/dev/null && ok "isaaclab import" || bad "isaaclab import 실패"

echo
echo "== 3. 내부 ROS 2 ${ROS_DISTRO} 라이브러리 =="
if [[ -n "${ISAAC_ROS_LIB_PATH:-}" ]]; then
  ok "경로: ${ISAAC_ROS_LIB_PATH}"
  [[ -f "${ISAAC_ROS_LIB_PATH}/librcutils.so" ]] \
    && ok "librcutils.so 있음" || bad "librcutils.so 없음"
else
  bad "내부 ROS 2 라이브러리 경로 탐색 실패"
fi

echo
echo "== 4. 시스템 ROS 2 ${ROS_DISTRO} (python 3.10) =="
ros-shell python3 -c "import rclpy, sys; print('  rclpy on python', '.'.join(map(str, sys.version_info[:2])))" 2>/dev/null \
  && ok "시스템 rclpy import" || bad "시스템 rclpy import 실패"

echo
echo "== 5. Isaac Lab 헤드리스 실행 (처음엔 셰이더 컴파일로 몇 분 걸립니다) =="
if [[ "${SKIP_SIM:-0}" == "1" ]]; then
  echo "  SKIP_SIM=1 이라 건너뜁니다"
else
  "${ISAACLAB_PATH}/isaaclab.sh" -p "${ISAACLAB_PATH}/scripts/tutorials/00_sim/create_empty.py" --headless \
    && ok "create_empty.py --headless" || bad "create_empty.py 실패"
fi

echo
[[ ${fail} -eq 0 ]] && echo "전부 통과" || echo "실패한 항목이 있습니다 (위 FAIL 참고)"
exit ${fail}
