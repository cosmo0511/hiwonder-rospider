# 단계별 설치 스크립트

`../Dockerfile` 은 전부 한 번에 빌드합니다. 이쪽은 **끊어서, 재실행 가능하게** 깔고 싶을 때 씁니다.
isaacsim 다운로드가 10GB 를 넘으니 뒤에서 깨지면 전부 날아갑니다 — 노트북에서는 이쪽이 안전합니다.

```
00_preflight.sh    설치 전 점검 (GPU·VRAM·RAM·디스크·GLIBC·네트워크). 아무것도 안 깝니다
10_conda_env.sh    miniconda + 가상환경 isaac_lab (python 3.11)      ← 여기서 시작
20_isaacsim.sh     isaacsim[all,extscache]==5.1.0                     ← 가장 오래 걸림
30_torch.sh        torch 2.7.0 / torchvision 0.22.0 (cu128)
40_isaaclab.sh     IsaacLab v2.3.1 클론 + ./isaaclab.sh -i + torch 재핀
50_verify.sh       import + 내부/시스템 ROS + 헤드리스 기동
run_all.sh         위를 순서대로
common.sh          공용 설정·로깅 (source 전용)
rtx4060.env        VRAM 8GB 급 설정값
```

## 쓰는 법

```bash
# 한 단계씩 (권장 — 어디서 깨졌는지 바로 보입니다)
bash docker/isaaclab/steps/00_preflight.sh
bash docker/isaaclab/steps/10_conda_env.sh
bash docker/isaaclab/steps/20_isaacsim.sh
bash docker/isaaclab/steps/30_torch.sh
bash docker/isaaclab/steps/40_isaaclab.sh
bash docker/isaaclab/steps/50_verify.sh

# 한 번에
bash docker/isaaclab/steps/run_all.sh
```

**끝난 단계는 건너뜁니다.** `/opt/.isaac-setup-state/` 에 마커를 남기므로, 중간에 깨져도
고치고 `run_all.sh` 를 다시 돌리면 거기서부터 이어집니다. 특정 단계를 다시 하려면 `FORCE=1`.

컨테이너 안에서 돌리는 걸 전제로 하지만, Ubuntu 22.04 머신에 직접 깔 때도 그대로 동작합니다
(3·4번 점검은 `/opt` 가 아니라 `ISAACLAB_PATH` 기준입니다).

## 바꿀 수 있는 것

모든 값은 환경변수로 덮어씁니다. 기본값은 `common.sh` 에 있습니다.

```bash
CONDA_ENV=myenv      bash steps/10_conda_env.sh    # 가상환경 이름
CONDA_DIR=~/miniconda3 bash steps/10_conda_env.sh  # conda 설치 위치
ISAACLAB_PATH=~/IsaacLab bash steps/40_isaaclab.sh # Isaac Lab 위치
SKIP_SIM=1           bash steps/50_verify.sh       # 기동 테스트 생략
FORCE=1              bash steps/10_conda_env.sh    # 환경 삭제 후 재생성
```

## conda 채널을 conda-forge 로 고정한 이유

conda 24.x 부터 Anaconda 기본 채널(`repo.anaconda.com/pkgs/main`, `/pkgs/r`)은 **ToS 동의**를
요구합니다. 동의하지 않은 상태로 `conda create` 를 무인 실행하면 이렇게 멈춥니다:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels.
```

도커 빌드든 스크립트든 똑같이 걸립니다. 게다가 기본 채널은 규모 있는 조직에 유상 라이선스를
요구합니다. 그래서 `10_conda_env.sh` 와 `Dockerfile` 모두 **conda-forge 단독**으로 환경을
만듭니다 (`-c conda-forge --override-channels`). Isaac Lab 은 파이썬 패키지를 전부 pip 로
설치하므로 채널 선택이 결과에 영향을 주지 않습니다.

기본 채널을 꼭 써야 한다면:

```bash
CONDA_CHANNEL=defaults ACCEPT_ANACONDA_TOS=1 bash steps/10_conda_env.sh
```

## RTX 4060 (8GB) 로 쓸 때

`00_preflight.sh` 가 VRAM 을 보고 경고를 띄웁니다. 요약하면:

| | |
|---|---|
| 학습 | `--headless` 필수. GUI 렌더러가 VRAM 3~4GB 를 먼저 먹습니다 |
| `--num_envs` | 기본값 4096 은 무조건 OOM. **32~64 에서 시작**해 두 배씩 올리세요 |
| `--enable_cameras` | env 당 VRAM 을 크게 먹습니다. 맨 마지막에 켜세요 |
| 시스템 RAM | 노트북은 16GB 가 흔한데 공식 최소는 32GB. 복잡한 씬은 VRAM 보다 RAM 이 먼저 터집니다 |
| 단편화 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — `10_conda_env.sh` 가 환경변수로 등록해 둡니다 |

```bash
source docker/isaaclab/steps/rtx4060.env

isaaclab -p ${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs ${ISAAC_NUM_ENVS}

# 다른 터미널에서 VRAM 보기
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2
```

OOM 메시지별 해석은 `rtx4060.env` 주석에 적어뒀습니다.

## 참고

- [Isaac Lab v2.3.1 — pip 설치](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html)
- [Isaac Sim 5.1 — 요구사양](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [RTX 4060 에서 Isaac Lab 돌린 후기](https://tech-multiverse.com/projects/yes-you-can-train-robots-in-isaac-lab-on-an-rtx-4060/)
- [IsaacLab #462 — OOM 인데 프로세스가 안 죽는 문제](https://github.com/isaac-sim/IsaacLab/issues/462)
- [IsaacLab #604 — 카메라 켜고 env 수 늘릴 때의 한계](https://github.com/isaac-sim/IsaacLab/issues/604)
- [IsaacLab #5350 — VRAM 보다 시스템 RAM 이 병목인 사례](https://github.com/isaac-sim/IsaacLab/issues/5350)
