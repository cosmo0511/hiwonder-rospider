#!/usr/bin/env bash
# steps/*.sh 공용 설정·로깅·중복실행 방지.
# source 전용. 모든 변수는 환경변수로 덮어쓸 수 있습니다.
#   예) CONDA_ENV=myenv bash steps/10_conda_env.sh

# ----------------------------------------------------------------- 버전 핀
export CONDA_ENV="${CONDA_ENV:-isaac_lab}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
export ISAACLAB_REF="${ISAACLAB_REF:-v2.3.1}"
export TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
export TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"
export EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.8}"

export ROS_DISTRO="${ROS_DISTRO:-humble}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# ----------------------------------------------------------------- 경로
export CONDA_DIR="${CONDA_DIR:-/opt/conda}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/opt/IsaacLab}"
export STATE_DIR="${STATE_DIR:-${ISAACLAB_PATH%/*}/.isaac-setup-state}"

# ----------------------------------------------------------------- EULA
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# ----------------------------------------------------------------- 출력
_c_red=$'\033[31m'; _c_grn=$'\033[32m'; _c_ylw=$'\033[33m'
_c_blu=$'\033[36m'; _c_bld=$'\033[1m';  _c_off=$'\033[0m'

step()  { printf '\n%s==> %s%s\n' "${_c_blu}${_c_bld}" "$*" "${_c_off}"; }
ok()    { printf '  %sOK%s   %s\n'   "${_c_grn}" "${_c_off}" "$*"; }
warn()  { printf '  %sWARN%s %s\n'   "${_c_ylw}" "${_c_off}" "$*"; }
bad()   { printf '  %sFAIL%s %s\n'   "${_c_red}" "${_c_off}" "$*"; }
info()  { printf '       %s\n' "$*"; }
die()   { bad "$*"; exit 1; }

# ----------------------------------------------- 중복 실행 방지 (재실행 가능하게)
# 각 단계는 끝날 때 마커를 남깁니다. FORCE=1 이면 마커를 무시하고 다시 돕니다.
mark_done()  { mkdir -p "${STATE_DIR}"; date -Iseconds > "${STATE_DIR}/$1"; }
is_done()    { [[ "${FORCE:-0}" != "1" && -f "${STATE_DIR}/$1" ]]; }
skip_if_done() {
  if is_done "$1"; then
    ok "$1 — 이미 완료 ($(cat "${STATE_DIR}/$1")). 다시 하려면 FORCE=1"
    exit 0
  fi
}

# --------------------------------------------------------- conda 환경 활성화
activate_env() {
  [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]] \
    || die "conda 가 없습니다 (${CONDA_DIR}). steps/10_conda_env.sh 를 먼저 돌리세요."
  # shellcheck disable=SC1091
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" \
    || die "'${CONDA_ENV}' 환경이 없습니다. steps/10_conda_env.sh 를 먼저 돌리세요."
  local v
  v="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [[ "${v}" == "${PYTHON_VERSION}" ]] \
    || die "python ${PYTHON_VERSION} 이어야 하는데 ${v} 입니다. Isaac Sim 5.x 는 3.11 고정입니다."
}
