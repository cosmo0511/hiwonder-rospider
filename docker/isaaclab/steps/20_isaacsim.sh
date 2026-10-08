#!/usr/bin/env bash
# 2단계 — Isaac Sim 5.1.0 (pip)
# 다운로드가 10GB 를 넘습니다. 가장 오래 걸리는 단계입니다.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./common.sh
skip_if_done 20_isaacsim
activate_env

step "GLIBC 재확인 (pip 설치 요구조건 2.35+)"
ldd --version | head -n1 | sed 's/^/       /'

step "isaacsim[all,extscache]==${ISAACSIM_VERSION} 설치"
info "10GB+ 다운로드입니다. 끊기면 이 스크립트를 그냥 다시 돌리면 이어서 받습니다."
pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com \
  || die "isaacsim 설치 실패"

step "검증"
python -c "import isaacsim; print('       isaacsim', getattr(isaacsim, '__version__', '(버전 속성 없음)'))" \
  || die "isaacsim import 실패"
ok "import 통과"
info "여기서는 import 만 봅니다. 실제 기동은 50_verify.sh 에서 확인합니다."

mark_done 20_isaacsim
echo
ok "완료 — steps/30_torch.sh 로 넘어가세요"
