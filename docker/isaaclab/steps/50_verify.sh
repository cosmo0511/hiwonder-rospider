#!/usr/bin/env bash
# 5단계 — 실제 기동까지 확인. 설치 자체는 건드리지 않으니 몇 번이고 돌려도 됩니다.
# 첫 실행은 셰이더 컴파일 때문에 몇 분 걸리고 그동안 멈춰 있는 것처럼 보입니다.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./common.sh
activate_env

fail=0

step "1. 버전 핀"
TORCH_VERSION="${TORCH_VERSION}" EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA}" python - <<'PY' && ok "python 3.11 / torch 핀 일치" || { bad "핀 어긋남"; fail=1; }
import os, sys, torch
print(f"       python {sys.version.split()[0]}   torch {torch.__version__} (cuda {torch.version.cuda})")
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.startswith(os.environ["TORCH_VERSION"]), torch.__version__
assert torch.version.cuda == os.environ["EXPECTED_TORCH_CUDA"], torch.version.cuda
PY

step "2. import"
python -c "import isaacsim" 2>/dev/null && ok "isaacsim" || { bad "isaacsim"; fail=1; }
python -c "import isaaclab, isaaclab_tasks" 2>/dev/null && ok "isaaclab, isaaclab_tasks" || { bad "isaaclab"; fail=1; }

step "3. 내부 ROS 2 ${ROS_DISTRO:-humble} 라이브러리"
if [[ -f /opt/isaaclab-scripts/isaac-env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/isaaclab-scripts/isaac-env.sh
  if [[ -n "${ISAAC_ROS_LIB_PATH:-}" ]]; then
    ok "${ISAAC_ROS_LIB_PATH}"
    [[ -f "${ISAAC_ROS_LIB_PATH}/librcutils.so" ]] \
      && ok "librcutils.so" || { bad "librcutils.so 없음"; fail=1; }
  else
    warn "내부 ROS 2 라이브러리 경로를 못 찾았습니다 (ROS 브리지 안 쓸 거면 무시)"
  fi
else
  info "isaac-env.sh 가 없습니다 (도커 이미지 밖에서 돌리는 중). 건너갑니다."
fi

step "4. 시스템 ROS 2 (python 3.10)"
if [[ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]]; then
  if command -v ros-shell >/dev/null; then
    ros-shell python3 -c "import rclpy, sys; print('       rclpy @ python', '.'.join(map(str, sys.version_info[:2])))" \
      && ok "시스템 rclpy" || { bad "시스템 rclpy import 실패"; fail=1; }
  else
    info "ros-shell 이 없습니다. 건너갑니다."
  fi
else
  info "/opt/ros 가 없습니다. 건너갑니다."
fi

step "5. 헤드리스 기동"
# 튜토리얼의 create_empty.py 는 무한 루프라 검증에 쓸 수 없습니다
# (`while simulation_app.is_running(): sim.step()` — 사용자가 끄기 전까지 안 끝남).
# 몇 스텝만 돌고 종료하는 _smoke_sim.py 를 씁니다.
if [[ "${SKIP_SIM:-0}" == "1" ]]; then
  info "SKIP_SIM=1 — 건너갑니다"
else
  info "첫 실행은 셰이더 컴파일로 5~10분 걸립니다. 로그가 멈춘 듯 보여도 정상입니다."
  info "최대 ${SIM_TIMEOUT:-1800}초까지 기다립니다."
  if timeout "${SIM_TIMEOUT:-1800}" \
       "${ISAACLAB_PATH}/isaaclab.sh" -p "$(pwd)/_smoke_sim.py" --headless --steps 60 \
       2>&1 | tee "${STATE_DIR}/smoke_sim.log" | grep -qE "\[SMOKE-OK\]"; then
    ok "Kit 기동 + 물리 60 스텝 완료"
  else
    rc=$?
    if [[ ${rc} -eq 124 ]]; then
      bad "시간 초과 (${SIM_TIMEOUT:-1800}초). 셰이더 컴파일이 더 필요하면 SIM_TIMEOUT 을 늘리세요."
    else
      bad "기동 실패 — 로그: ${STATE_DIR}/smoke_sim.log"
    fi
    fail=1
  fi
fi

echo
if (( fail )); then
  bad "실패 항목이 있습니다. docker/isaaclab/README.md 의 트러블슈팅 표를 보세요."
  exit 1
fi
mark_done 50_verify
ok "전부 통과 — 설치 끝났습니다"
info "VRAM 8GB 급이면 학습 실행 전에 steps/rtx4060.env 를 읽어보세요."
