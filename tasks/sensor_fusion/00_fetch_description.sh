#!/usr/bin/env bash
# RosPider 로봇 기술(description) 패키지를 레포 안으로 가져온다.
#
# Hiwonder 가 공개한 ROS 2 워크스페이스(github.com/Hiwonder/ROSpider) 안에
# src/simulations/rospider_description/ 가 있고, 거기에 xacro 와 STL 메시가 전부 들어 있다.
# 우리는 그 디렉터리만 뽑아 assets/ 아래에 둔다. (약 20 MB, STL 36개)
#
# 메시와 xacro 는 Hiwonder 저작물이다. 레포에 커밋할지, 이 스크립트로 매번
# 받아올지는 라이선스를 확인하고 정하자. 기본은 "받아오기"(git-ignore 대상).

set -u  # set -e 는 쓰지 않는다. CLAUDE.md 참고: 실패하는 명령치환이 쉘을 죽인다.

REPO_URL="${REPO_URL:-https://github.com/Hiwonder/ROSpider}"
DEST="${DEST:-assets/rospider_description}"
SRC_SUBDIR="src/simulations/rospider_description"
TMP_DIR="$(mktemp -d)"

echo "[1/3] ${REPO_URL} 에서 ${SRC_SUBDIR} 만 받는다 (sparse checkout)"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${TMP_DIR}/repo"
if [ $? -ne 0 ]; then
  echo "[실패] clone 이 안 됐다. 네트워크나 URL 을 확인하자." >&2
  rm -rf "${TMP_DIR}"
  exit 1
fi
git -C "${TMP_DIR}/repo" sparse-checkout set "${SRC_SUBDIR}"

echo "[2/3] ${DEST} 로 복사"
mkdir -p "${DEST}"
cp -r "${TMP_DIR}/repo/${SRC_SUBDIR}/urdf" "${TMP_DIR}/repo/${SRC_SUBDIR}/meshes" "${DEST}/"
rm -rf "${TMP_DIR}"

echo "[3/3] 확인"
echo "  xacro  : $(find "${DEST}/urdf" -name '*.xacro' | wc -l) 개"
echo "  메시   : $(find "${DEST}/meshes" -iname '*.stl' | wc -l) 개"
echo
echo "다음 단계:"
echo "  pip install xacro        # 한 번만"
echo "  python tasks/sensor_fusion/01_xacro_to_urdf.py --pkg ${DEST}"
