#!/usr/bin/env bash
# 0 → 5 단계를 순서대로. 이미 끝난 단계는 건너뜁니다.
# 중간에 깨져도 고치고 다시 돌리면 거기서부터 이어집니다.
#
#   bash steps/run_all.sh
#   SKIP_SIM=1 bash steps/run_all.sh     # 마지막 기동 테스트 생략
#   FORCE=1 bash steps/run_all.sh        # 전부 처음부터
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./common.sh

for s in 00_preflight 10_conda_env 20_isaacsim 30_torch 40_isaaclab 50_verify; do
  printf '\n%s┌─ %s %s\n' "${_c_bld}" "${s}" "${_c_off}"
  bash "./${s}.sh" || die "${s} 에서 멈췄습니다. 고친 뒤 'bash steps/run_all.sh' 로 이어서 돌리세요."
done

echo
ok "전체 완료"
