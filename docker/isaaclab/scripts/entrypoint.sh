#!/usr/bin/env bash
# 기본 진입점: Isaac 쪽 환경을 깔고 넘겨받은 명령을 실행합니다.
# ROS 노드를 돌릴 때는 컨테이너 안에서 `ros-shell` 로 갈아타세요.
#
# `set -e` 를 쓰지 않습니다. 환경 준비 중 조회 하나가 실패했다고(설치 전이라 python 이
# 없다든지) 컨테이너가 통째로 죽으면 안 되기 때문입니다.
source /opt/isaaclab-scripts/isaac-env.sh || true
exec "$@"
