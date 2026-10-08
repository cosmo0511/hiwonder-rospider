#!/usr/bin/env bash
# 3단계 — PyTorch 2.7.0 + torchvision 0.22.0 (cu128)
# isaacsim 이 끌고 온 torch 를 지정 버전으로 덮어씁니다.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./common.sh
skip_if_done 30_torch
activate_env

step "현재 torch"
python - <<'PY' || true
try:
    import torch
    print(f"       {torch.__version__}  (cuda {torch.version.cuda})")
except ImportError:
    print("       (없음)")
PY

step "torch==${TORCH_VERSION} / torchvision==${TORCHVISION_VERSION} (${TORCH_CUDA_TAG})"
# isaacsim 이 PyPI 기본 휠(cu126)을 끌고 오므로 여기서 cu128 로 덮어씁니다.
# nvidia-* 보조 휠도 12.6 -> 12.8 로 같이 갈립니다. 2~3GB 받습니다.
pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX}" \
  || die "torch 설치 실패"

# torchaudio 는 공식 설치 문서에 없지만 isaacsim 이 의존성으로 끌고 옵니다.
# torch 만 cu128 로 바꾸면 torchaudio 는 cu126 으로 남아 섞이고,
# 무언가 import 하는 순간 undefined symbol 로 터집니다. 있을 때만 맞춰 줍니다.
if pip show torchaudio >/dev/null 2>&1; then
  step "torchaudio 를 같은 빌드로 정렬"
  installed_ta="$(python -c 'import importlib.metadata as m; print(m.version("torchaudio"))' 2>/dev/null || echo "?")"
  info "현재: ${installed_ta}"
  if pip install -U "torchaudio==${TORCH_VERSION}" --index-url "${TORCH_INDEX}" 2>/dev/null; then
    ok "torchaudio ${TORCH_VERSION} (${TORCH_CUDA_TAG})"
  else
    warn "${TORCH_CUDA_TAG} 인덱스에 torchaudio==${TORCH_VERSION} 이 없습니다."
    info "  Isaac Lab 은 torchaudio 를 쓰지 않으므로 설치는 계속합니다."
    info "  나중에 torchaudio import 에서 undefined symbol 이 나면 제거하세요:  pip uninstall -y torchaudio"
  fi
fi

step "검증"
TORCH_VERSION="${TORCH_VERSION}" EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA}" python - <<'PY'
import os, sys, torch
want, want_cuda = os.environ["TORCH_VERSION"], os.environ["EXPECTED_TORCH_CUDA"]
print(f"       torch        {torch.__version__}")
print(f"       built cuda   {torch.version.cuda}")
print(f"       available    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"       GPU          {torch.cuda.get_device_name(0)}  "
          f"({total // 1024**2} MiB, {free // 1024**2} MiB 여유)")
assert torch.__version__.startswith(want), f"{torch.__version__} != {want}"
assert torch.version.cuda == want_cuda, f"{torch.version.cuda} != {want_cuda}"
PY
ok "핀 일치"

# torch 는 cu128 인데 nvidia-* 보조 휠이 12.6 으로 남아 있으면 런타임에 터집니다.
mixed="$(pip list --format=freeze 2>/dev/null | grep -E '^nvidia-.*-cu12==12\.6\.' || true)"
if [[ -n "${mixed}" ]]; then
  warn "CUDA 12.6 보조 휠이 남아 있습니다:"
  echo "${mixed}" | sed 's/^/         /'
  info "  보통은 torch 가 번들한 것이 우선이라 문제없지만, 런타임 심볼 에러가 나면"
  info "  'pip install -U torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} --force-reinstall' 로 다시 깔아보세요."
fi

# cuda.is_available() 이 False 면 뒤 단계가 전부 무의미하니 여기서 끊습니다.
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch 가 GPU 를 못 봅니다. 컨테이너면 --gpus all / nvidia-container-toolkit 을 확인하세요."
ok "GPU 접근 가능"

mark_done 30_torch
echo
ok "완료 — steps/40_isaaclab.sh 로 넘어가세요"
