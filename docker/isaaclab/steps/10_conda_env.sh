#!/usr/bin/env bash
# 1단계 — miniconda 설치 + 가상환경 `isaac_lab` (python 3.11) 생성
#
#   bash steps/10_conda_env.sh
#   FORCE=1 bash steps/10_conda_env.sh      # 환경을 지우고 다시 만듭니다
#   CONDA_ENV=other bash steps/10_conda_env.sh
#
# python 3.11 은 Isaac Sim 5.x 의 고정 요구사항입니다. 3.10/3.12 는 동작하지 않습니다.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./common.sh
skip_if_done 10_conda_env

step "miniconda 확인"
if [[ -x "${CONDA_DIR}/bin/conda" ]]; then
  ok "이미 있음: ${CONDA_DIR} ($("${CONDA_DIR}/bin/conda" --version))"
else
  info "${CONDA_DIR} 에 설치합니다"
  installer="$(mktemp -t miniconda-XXXXXX.sh)"
  trap 'rm -f "${installer}"' EXIT

  arch="$(uname -m)"
  case "${arch}" in
    x86_64)  mc_file="Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64) mc_file="Miniconda3-latest-Linux-aarch64.sh" ;;
    *)       die "지원하지 않는 아키텍처: ${arch}" ;;
  esac

  curl -fsSL "https://repo.anaconda.com/miniconda/${mc_file}" -o "${installer}" \
    || die "miniconda 다운로드 실패"
  # -b 무인설치, -p 설치경로, -u 는 "기존 설치 갱신".
  # ★ -u 가 꼭 필요합니다. ${CONDA_DIR} 가 도커 named volume 의 마운트 지점이면
  #   내용이 비어 있어도 디렉터리는 이미 존재하고, 설치기가 이렇게 거부합니다:
  #     ERROR: File or directory already exists: '/opt/conda'
  #   마운트 지점이라 rmdir 도 안 되므로 -u 로 그 안에 설치합니다.
  #   디렉터리가 아예 없을 때도 -u 는 문제없이 동작합니다.
  bash "${installer}" -b -u -p "${CONDA_DIR}" || die "miniconda 설치 실패"
  ok "설치 완료: $("${CONDA_DIR}/bin/conda" --version)"
fi

conda_bin="${CONDA_DIR}/bin/conda"

step "conda 기본 설정"
"${conda_bin}" config --set always_yes true
# base 자동 활성화를 끕니다. 켜져 있으면 ros-shell 쪽에서 base 의 libstdc++ 가 섞입니다.
"${conda_bin}" config --set auto_activate_base false

# conda 24.x 이후로는 Anaconda 기본 채널(repo.anaconda.com/pkgs/*)에 ToS 동의가 필요하고,
# 동의 없이 무인 실행하면 CondaToSNonInteractiveError 로 멈춥니다.
# 게다가 Anaconda 기본 채널은 규모가 있는 조직에 유상 라이선스를 요구합니다.
# 그래서 기본 채널을 아예 쓰지 않고 conda-forge 만으로 갑니다. ToS 문제 자체가 사라집니다.
#   기본 채널을 꼭 써야 한다면: CONDA_CHANNEL=defaults ACCEPT_ANACONDA_TOS=1
export CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"
if [[ "${CONDA_CHANNEL}" == "conda-forge" ]]; then
  "${conda_bin}" config --add channels conda-forge 2>/dev/null || true
  "${conda_bin}" config --remove channels defaults 2>/dev/null || true
  ok "채널: conda-forge 단독 (Anaconda ToS 불필요)"
else
  if [[ "${ACCEPT_ANACONDA_TOS:-0}" == "1" ]]; then
    for ch in main r; do
      "${conda_bin}" tos accept --override-channels \
        --channel "https://repo.anaconda.com/pkgs/${ch}" >/dev/null 2>&1 || true
    done
    ok "Anaconda 기본 채널 ToS 동의 처리"
  else
    warn "CONDA_CHANNEL=${CONDA_CHANNEL} 인데 ToS 동의가 없습니다."
    info "  ACCEPT_ANACONDA_TOS=1 을 주거나 CONDA_CHANNEL=conda-forge 로 두세요."
  fi
fi
ok "always_yes=true, auto_activate_base=false"

step "가상환경 '${CONDA_ENV}' (python ${PYTHON_VERSION})"
if "${conda_bin}" env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  if [[ "${FORCE:-0}" == "1" ]]; then
    warn "FORCE=1 — 기존 '${CONDA_ENV}' 환경을 삭제합니다"
    "${conda_bin}" env remove -n "${CONDA_ENV}" -y
    "${conda_bin}" create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" \
      -c "${CONDA_CHANNEL}" --override-channels
  else
    ok "이미 존재합니다 (다시 만들려면 FORCE=1)"
  fi
else
  "${conda_bin}" create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" \
      -c "${CONDA_CHANNEL}" --override-channels \
    || die "환경 생성 실패"
  ok "생성 완료"
fi

step "환경 활성화 시 자동으로 적용될 변수 등록"
# conda activate isaac_lab 만 해도 EULA 동의와 ROS 설정이 따라오게 합니다.
# (Isaac Sim 은 OMNI_KIT_ACCEPT_EULA 가 없으면 무인 기동 시 멈춥니다.)
"${conda_bin}" env config vars set -n "${CONDA_ENV}" \
  OMNI_KIT_ACCEPT_EULA=YES \
  ACCEPT_EULA=Y \
  PRIVACY_CONSENT=Y \
  ISAACLAB_PATH="${ISAACLAB_PATH}" \
  RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  >/dev/null
# VRAM 8GB 급에서 단편화로 인한 OOM 을 줄여줍니다. 넉넉한 GPU 에서도 해롭지 않습니다.
"${conda_bin}" env config vars set -n "${CONDA_ENV}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True >/dev/null
ok "OMNI_KIT_ACCEPT_EULA, ISAACLAB_PATH, PYTORCH_CUDA_ALLOC_CONF 등 등록"

step "검증"
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python -m pip install --upgrade pip >/dev/null
actual="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "${actual}" == "${PYTHON_VERSION}" ]] \
  || die "python ${PYTHON_VERSION} 이 아니라 ${actual} 입니다"
ok "python $(python -V 2>&1 | cut -d' ' -f2)  @  ${CONDA_PREFIX}"
ok "pip    $(pip --version | cut -d' ' -f2)"

mark_done 10_conda_env
echo
ok "완료 — steps/20_isaacsim.sh 로 넘어가세요"
info "쉘에서 직접 쓰려면:  source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
