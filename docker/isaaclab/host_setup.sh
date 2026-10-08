#!/usr/bin/env bash
# 호스트(노트북)에 Docker + NVIDIA Container Toolkit 을 깝니다. 최초 1회.
#
#   bash docker/isaaclab/host_setup.sh
#
# 왜 스크립트인가: nvidia-container-toolkit 은 우분투 기본 저장소에 없습니다.
# NVIDIA 저장소를 먼저 등록해야 하고, apt 는 패키지 하나라도 못 찾으면 전부 취소하므로
# `apt install docker.io nvidia-container-toolkit` 같은 한 줄은 아무것도 안 깝니다.
#
# 이 스크립트는 몇 번을 돌려도 안전합니다 (이미 된 건 건너뜁니다).
set -eo pipefail

_r=$'\033[31m'; _g=$'\033[32m'; _y=$'\033[33m'; _b=$'\033[36m'; _B=$'\033[1m'; _0=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "${_b}${_B}" "$*" "${_0}"; }
ok()   { printf '  %sOK%s   %s\n' "${_g}" "${_0}" "$*"; }
warn() { printf '  %sWARN%s %s\n' "${_y}" "${_0}" "$*"; }
die()  { printf '  %sFAIL%s %s\n' "${_r}" "${_0}" "$*"; exit 1; }

# root 로 돌면 $USER 가 root 라 docker 그룹 추가가 엉뚱한 계정에 붙습니다.
# sudo 로 들어온 거면 원래 계정을 찾아 쓰고, 순수 root 면 그룹 단계만 건너뜁니다.
TARGET_USER="${SUDO_USER:-${USER:-$(id -un)}}"
if [[ "${EUID}" -eq 0 ]]; then
  if [[ -n "${SUDO_USER:-}" ]]; then
    printf '  %sWARN%s sudo 로 실행 중입니다. docker 그룹은 %s 에 추가합니다.\n' \
      "${_y}" "${_0}" "${SUDO_USER}"
  else
    printf '  %sWARN%s root 로 실행 중입니다. docker 그룹 추가는 건너뜁니다.\n' "${_y}" "${_0}"
    TARGET_USER=""
  fi
fi

. /etc/os-release
step "호스트 확인"
ok "${PRETTY_NAME}"
[[ "${ID}" == "ubuntu" ]] || warn "우분투가 아닙니다 (${ID}). 아래 명령이 그대로 안 맞을 수 있습니다."

step "1. NVIDIA 드라이버"
command -v nvidia-smi >/dev/null || die "nvidia-smi 가 없습니다. 드라이버부터 설치하세요: sudo ubuntu-drivers install"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/       /'
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
awk -v d="${drv%%.*}" 'BEGIN{exit !(d+0 >= 570)}' \
  && ok "드라이버 ${drv} — CUDA 12.8 요구선(570+) 충족" \
  || die "드라이버 ${drv} — 570 이상이 필요합니다"

step "2. Docker"
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  ok "이미 동작 중: $(docker --version)"
elif command -v docker >/dev/null; then
  warn "docker 는 있는데 데몬이 안 돕니다. 서비스를 올립니다."
  sudo systemctl enable --now docker
else
  if snap list docker >/dev/null 2>&1; then
    warn "snap 으로 깐 docker 가 있습니다. snap 버전은 /dev/nvidia* 에 접근하지 못하는 경우가 있습니다."
    warn "  sudo snap remove docker  후 이 스크립트를 다시 돌리길 권합니다."
  fi
  # 우분투 저장소의 docker.io 는 배포판에 따라 compose v2 가 없습니다.
  # 버전 상관없이 동일하게 가도록 Docker 공식 저장소를 씁니다.
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" \
    | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                          docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
  ok "설치 완료: $(docker --version)"
fi
docker compose version >/dev/null 2>&1 \
  && ok "compose: $(docker compose version --short)" \
  || die "'docker compose' 가 없습니다. docker-compose-plugin 을 설치하세요."

step "3. NVIDIA Container Toolkit"
# ★ 여기가 핵심. 이 패키지는 우분투 기본 저장소에 없고 NVIDIA 저장소에 있습니다.
if command -v nvidia-ctk >/dev/null; then
  ok "이미 있음: $(nvidia-ctk --version | head -n1)"
else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  ok "설치 완료: $(nvidia-ctk --version | head -n1)"
fi

step "4. Docker 에 NVIDIA 런타임 연결"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
ok "연결 + 데몬 재시작"

step "5. docker 그룹"
need_relogin=0
if [[ -z "${TARGET_USER}" ]]; then
  ok "root 실행 — 건너뜁니다"
elif id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx docker; then
  ok "${TARGET_USER} 는 이미 docker 그룹입니다"
else
  sudo usermod -aG docker "${TARGET_USER}"
  warn "${TARGET_USER} 를 docker 그룹에 넣었습니다 — 로그아웃/재로그인 해야 적용됩니다"
  need_relogin=1
fi

step "6. 컨테이너가 GPU 를 보는지 확인"
# 그룹 적용 전이면 sudo 로 확인합니다.
dk=(docker); (( need_relogin )) && dk=(sudo docker)
if "${dk[@]}" run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi \
     --query-gpu=name,memory.total --format=csv,noheader; then
  ok "컨테이너에서 GPU 가 보입니다"
else
  die "컨테이너가 GPU 를 못 봅니다. 위 에러를 확인하세요."
fi

step "7. GUI (X11)"
if command -v xhost >/dev/null; then
  xhost +local:root >/dev/null && ok "xhost +local:root"
else
  warn "xhost 가 없습니다. GUI 를 띄울 거면: sudo apt install -y x11-xserver-utils"
fi

echo
ok "호스트 준비 끝"
if (( need_relogin )); then
  echo
  printf '  %s한 번 로그아웃했다 다시 로그인하세요.%s (안 하면 docker 명령마다 sudo 가 필요합니다)\n' "${_B}" "${_0}"
  printf '  지금 바로 쓰려면 이 터미널에서:  %snewgrp docker%s\n' "${_B}" "${_0}"
fi
echo
echo "  다음:  docker compose -f docker/isaaclab/docker-compose.yml build base"
