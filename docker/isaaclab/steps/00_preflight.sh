#!/usr/bin/env bash
# 0단계 — 설치 전 점검. 여기서 걸리는 건 뒤에서 반드시 터집니다.
# 설치는 하지 않습니다. 몇 번이고 돌려도 됩니다.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./common.sh

fail=0

step "1. OS / GLIBC"
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  info "${PRETTY_NAME:-unknown}"
  case "${VERSION_ID:-}" in
    22.04) ok "Ubuntu 22.04 — ROS 2 Humble 과 맞는 조합입니다" ;;
    24.04) warn "Ubuntu 24.04 — Isaac Sim 5.1 은 지원하지만 apt ROS 2 는 Jazzy(python 3.12) 입니다."
           info "  Humble 을 쓰려면 22.04 여야 합니다." ;;
    *)     warn "Ubuntu ${VERSION_ID:-?} — Isaac Sim 5.1 이 지원하는 건 22.04 / 24.04 입니다." ;;
  esac
fi
glibc="$(ldd --version | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)"
info "GLIBC ${glibc}"
awk -v g="${glibc}" 'BEGIN{exit !(g+0 >= 2.35)}' \
  && ok "GLIBC 2.35+ (isaacsim pip 설치 요구조건)" \
  || { bad "GLIBC 2.35 미만 — isaacsim pip 설치가 불가합니다"; fail=1; }

step "2. GPU / 드라이버"
if ! command -v nvidia-smi >/dev/null; then
  bad "nvidia-smi 가 없습니다"
  info "컨테이너 안이면 --gpus all + nvidia-container-toolkit 을 확인하세요."
  fail=1
else
  nvidia-smi --query-gpu=name,memory.total,driver_version \
             --format=csv,noheader 2>/dev/null | sed 's/^/       /' || true
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"

  # RT 코어 없는 데이터센터 GPU 는 Isaac Sim 미지원
  if grep -qiE 'A100|H100|V100|A30|A40G?$|T4' <<<"${gpu}"; then
    bad "${gpu} 는 RT 코어가 없어 Isaac Sim 이 지원하지 않습니다"; fail=1
  else
    ok "RT 코어 있는 GPU 로 보입니다"
  fi

  if   (( vram >= 16000 )); then ok "VRAM ${vram}MiB — 공식 권장 충족"
  elif (( vram >=  7000 )); then
    warn "VRAM ${vram}MiB — 공식 최소는 RTX 4080 16GB 입니다. 돌아가긴 하지만:"
    info "  · 학습은 --headless 필수 (GUI 가 VRAM 3~4GB 를 먹습니다)"
    info "  · --num_envs 를 32~64 에서 시작해 두 배씩 올리며 한계를 찾으세요"
    info "  · --enable_cameras 는 VRAM 을 크게 먹으니 마지막에 켜세요"
    info "  · 자세한 건 steps/rtx4060.env 참고"
  else
    bad "VRAM ${vram}MiB — Isaac Sim 기동조차 어렵습니다"; fail=1
  fi

  awk -v d="${drv%%.*}" 'BEGIN{exit !(d+0 >= 570)}' \
    && ok "드라이버 ${drv} (CUDA 12.8 요구선 570+ 충족)" \
    || { bad "드라이버 ${drv} — CUDA 12.8 에는 570 이상이 필요합니다 (5.1 테스트 버전 580.65.06)"; fail=1; }
fi

step "3. 시스템 RAM"
# MemTotal 은 커널이 쓰는 분량이 빠진 값이라 16GiB 머신도 15.x 로 나옵니다. 올림 처리.
ram_mb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 ))
ram_gb=$(( (ram_mb + 1023) / 1024 ))
info "${ram_gb} GB (MemTotal ${ram_mb} MiB)"
# 16GB 장착 노트북도 MemTotal 은 15000~15800 MiB 로 잡힙니다(iGPU·펌웨어 예약).
# 그래서 "16GB 이상" 을 GiB 로 비교하면 멀쩡한 머신이 FAIL 납니다. MiB 로 봅니다.
if   (( ram_mb >= 30000 )); then ok "32GB 급 (공식 최소 충족)"
elif (( ram_mb >= 14000 )); then
  warn "16GB 급 — 공식 최소는 32GB 입니다. 돌아가지만 복잡한 씬에서 VRAM 보다 먼저 터집니다."
  info "  스왑을 24GB 쯤 잡아두면 기동 실패가 크게 줄어듭니다."
  info "  컨테이너가 아니라 **호스트에서** 잡아야 합니다:"
  info "    sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile"
  info "    sudo mkswap /swapfile && sudo swapon /swapfile"
  info "    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
else
  bad "${ram_gb}GB — 부족합니다"; fail=1
fi
sw_mb=$(( $(awk '/SwapTotal/{print $2}' /proc/meminfo) / 1024 ))
if (( sw_mb >= 8000 )); then ok "스왑 ${sw_mb} MiB"
elif (( ram_mb < 30000 )); then warn "스왑 ${sw_mb} MiB — RAM 이 빠듯한데 스왑도 적습니다"; fi

step "4. 디스크 여유"
# df -P 와 --output 은 같이 못 씁니다. -Pk + awk 가 busybox 에서도 돕니다.
avail_kb="$(df -Pk "${ISAACLAB_PATH%/*}" 2>/dev/null | awk 'NR==2{print $4}')"
avail_gb=$(( ${avail_kb:-0} / 1024 / 1024 ))
info "${ISAACLAB_PATH%/*} 에 ${avail_gb} GB"
if   (( avail_gb >= 100 )); then ok "100GB+"
elif (( avail_gb >=  60 )); then warn "${avail_gb}GB — isaacsim[all,extscache] 만 10GB 넘고 캐시가 더 붙습니다. 빠듯합니다."
else bad "${avail_gb}GB — 부족합니다. 100GB 를 권합니다."; fail=1; fi

step "5. 네트워크"
# 루트(/)는 403 을 주는 호스트가 있어서(download.pytorch.org 는 S3 버킷) 실제로
# 설치에 쓰는 경로를 찍습니다. 루트로 확인하면 멀쩡한 네트워크가 FAIL 로 나옵니다.
for u in "https://pypi.nvidia.com/isaacsim/" \
         "${TORCH_INDEX}/torch/" \
         "https://github.com/isaac-sim/IsaacLab" \
         "https://repo.anaconda.com/miniconda/"; do
  host="${u#https://}"; host="${host%%/*}"
  if curl -fsSL --max-time 15 -o /dev/null "${u}" 2>/dev/null \
  || curl -fsSI --max-time 15 -o /dev/null "${u}" 2>/dev/null; then
    ok "${host}"
  else
    bad "${host} 접근 실패  (${u})"; fail=1
  fi
done

echo
if (( fail )); then
  bad "FAIL 항목을 먼저 해결하세요. 그대로 진행하면 설치 중간에 터집니다."
  exit 1
fi
ok "점검 통과 — steps/10_conda_env.sh 로 넘어가세요"
