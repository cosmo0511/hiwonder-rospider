#!/usr/bin/env bash
# ROS 2 Humble 쪽 쉘 환경. 시스템 python 3.10 을 씁니다.
#   - conda 를 비활성화하고 /opt/ros/humble 을 source
#   - conda 의 libstdc++ 가 섞이면 RMW 가 GLIBCXX_3.4.30 not found 로 죽습니다
#
# source 전용.

# conda 가 활성화돼 있으면 모두 빠져나옵니다.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_DIR:-/opt/conda}/etc/profile.d/conda.sh"
  while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate || break; done
fi

# conda 가 끼워 넣은 라이브러리 경로를 걷어냅니다.
unset LD_LIBRARY_PATH
export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v '^/opt/conda' | paste -sd: -)"

# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if [[ -f /workspace/ros2_ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /workspace/ros2_ws/install/setup.bash
fi
