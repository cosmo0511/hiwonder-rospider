# 터미널에 칠 명령 — 순서대로

복붙하면 됩니다. 각 블록이 끝나야 다음으로 갑니다.

- **0 ~ 2단계는 노트북(호스트)** 터미널
  (`host_setup.sh` 는 레포 안에 있으니 **1단계 클론을 먼저** 하고 0단계를 돌려도 됩니다)
- **3단계부터는 컨테이너 안** — 프롬프트가 `root@...:/workspace#` 로 바뀝니다

---

## 0단계 · 호스트 준비 (최초 1회)

### Ubuntu 를 쓰는 경우

```bash
# 드라이버 확인. RTX 4060 이 보이고 Driver Version 이 570 이상이어야 합니다.
nvidia-smi

# 레포를 받고 (1단계) 나서
bash docker/isaaclab/host_setup.sh
```

`host_setup.sh` 가 Docker + NVIDIA Container Toolkit 설치, 런타임 연결, docker 그룹,
GPU 확인까지 다 합니다. 여러 번 돌려도 안전합니다.

> **`apt install nvidia-container-toolkit` 한 줄로는 안 됩니다.**
> 이 패키지는 우분투 기본 저장소에 없고 NVIDIA 저장소에 있습니다.
> 게다가 apt 는 패키지 하나라도 못 찾으면 **아무것도 설치하지 않고 전부 취소**합니다.
> 그래서 `apt install docker.io ... nvidia-container-toolkit` 는 docker 까지 같이 날아가고,
> 이어지는 `nvidia-ctk: command not found` / `docker.service not found` 로 번집니다.

직접 치고 싶다면 이 순서입니다 (스크립트가 하는 일):

```bash
# 1) Docker 공식 저장소 — 우분투 저장소의 docker.io 는 배포판에 따라 compose v2 가 없습니다
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 2) NVIDIA Container Toolkit 저장소 ★ 빠뜨리면 위 에러가 납니다
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3) Docker 에 연결
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker $USER        # 로그아웃/재로그인 필요 (급하면 newgrp docker)

# 4) 확인 — 여기서 nvidia-smi 출력이 나와야 다음으로 갑니다
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

# 5) GUI 쓸 거면
xhost +local:root
```

### Windows 11 을 쓰는 경우

PowerShell(관리자)에서:

```powershell
wsl --install -d Ubuntu-22.04
```

그리고 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 을 설치하고
**Settings → Resources → WSL Integration** 에서 Ubuntu-22.04 를 켭니다.
드라이버는 **Windows 쪽에** 설치합니다 (WSL 안에 리눅스 드라이버를 깔면 안 됩니다).

이후 모든 명령은 WSL 우분투 터미널에서 칩니다:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

> Windows 는 WSL2 메모리 상한이 기본으로 호스트 RAM 의 절반입니다.
> 16GB 노트북이면 8GB 만 잡혀서 Isaac Sim 이 기동 중에 죽습니다.
> `C:\Users\<이름>\.wslconfig` 를 만들어 두세요:
> ```ini
> [wsl2]
> memory=12GB
> swap=24GB
> ```
> 그리고 PowerShell 에서 `wsl --shutdown` 후 다시 열면 적용됩니다.

---

## 1단계 · 레포 받기

```bash
git clone https://github.com/cosmo0511/hiwonder-rospider.git
cd hiwonder-rospider

```

---

## 2단계 · 설치용 컨테이너 빌드 (10분 내외)

시스템 패키지 + ROS 2 Humble 만 들어간 가벼운 이미지입니다.
Isaac Sim 은 여기 안 들어 있습니다 — 그건 컨테이너 안에서 단계별로 깝니다.

```bash
docker compose -f docker/isaaclab/docker-compose.yml build base
```

컨테이너 안으로 들어갑니다:

```bash
docker compose -f docker/isaaclab/docker-compose.yml run --rm base
```

들어가면 이런 메시지가 뜹니다 — **정상입니다.** 아직 conda 를 안 깔았으니까요.

```
[isaac-env] conda 가 아직 없습니다 (/opt/conda).
[isaac-env] 설치를 시작하려면:  bash /opt/isaaclab-steps/00_preflight.sh
```

---

## 3단계 · 컨테이너 안에서 설치

여기서부터는 **컨테이너 안** 입니다. 한 줄씩, 끝나는 걸 보고 다음으로 갑니다.

```bash
# 점검만 합니다. 아무것도 안 깔아요. FAIL 이 있으면 여기서 멈추고 해결하세요.
bash /opt/isaaclab-steps/00_preflight.sh

# 가상환경 isaac_lab (python 3.11)          — 1분
bash /opt/isaaclab-steps/10_conda_env.sh

# Isaac Sim 5.1.0                            — 30~60분, 10GB+ 다운로드
bash /opt/isaaclab-steps/20_isaacsim.sh

# PyTorch 2.7.0 (cu128)                      — 5분
bash /opt/isaaclab-steps/30_torch.sh

# Isaac Lab 2.3.1                            — 10분
bash /opt/isaaclab-steps/40_isaaclab.sh

# 검증 (실제 기동까지)                        — 첫 실행은 셰이더 컴파일로 수 분
bash /opt/isaaclab-steps/50_verify.sh
```

한 번에 가려면:

```bash
bash /opt/isaaclab-steps/run_all.sh
```

### 중간에 끊겼으면

**그냥 똑같은 명령을 다시 치면 됩니다.** 끝난 단계는 건너뜁니다.
컨테이너를 닫았어도 설치 결과는 도커 볼륨(`isaac-conda`, `isaac-lab`)에 남아 있습니다.

```bash
# 호스트에서 다시 들어가기
docker compose -f docker/isaaclab/docker-compose.yml run --rm base

# 컨테이너 안에서 이어서
bash /opt/isaaclab-steps/run_all.sh
```

---

## 4단계 · 써보기 (컨테이너 안)

설치 스크립트는 각각 별도 프로세스로 돌았으니, 지금 쉘에는 conda 환경이 안 잡혀 있습니다.
먼저 한 줄 치세요 (`exit` 후 다시 들어가도 됩니다):

```bash
source /opt/isaaclab-scripts/isaac-env.sh
```

프롬프트에 `(isaac_lab)` 이 붙으면 준비됐습니다.

```bash
# 빈 씬 띄우기 — 제일 가벼운 확인
isaaclab -p ${ISAACLAB_PATH}/scripts/tutorials/00_sim/create_empty.py --headless

# RTX 4060 (8GB) 학습 설정 적용
source /opt/isaaclab-steps/rtx4060.env

# 학습. num_envs 는 64 에서 시작합니다 (기본값 4096 은 8GB 에서 무조건 OOM)
isaaclab -p ${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs ${ISAAC_NUM_ENVS}
```

VRAM 을 보려면 **호스트에서** 다른 터미널을 열고:

```bash
watch -n 2 nvidia-smi
```

여유가 남으면 `--num_envs 128` → `256` 으로 올려보세요. OOM 나면 직전 값이 상한입니다.

### ROS 2 를 쓸 때

같은 컨테이너 안에서 쉘만 갈아탑니다. `isaac-shell` 안에서 ROS 를 source 하면 안 됩니다
(python 3.10/3.11 충돌로 Isaac Sim 기동이 깨집니다 — `README.md` 참고).

```bash
ros-shell                  # ROS 2 쪽 쉘 (시스템 python 3.10)
ros2 topic list
exit                       # 돌아오기
```

---

## 자주 걸리는 것

| 터미널에 뜨는 것 | 할 일 |
|---|---|
| `docker: permission denied` | `sudo usermod -aG docker $USER` 후 로그아웃/재로그인 |
| `E: Unable to locate package nvidia-container-toolkit` | NVIDIA 저장소 미등록. `bash docker/isaaclab/host_setup.sh` |
| `nvidia-ctk: command not found` | 위와 같은 원인 (apt 가 통째로 취소됐습니다) |
| `Unit docker.service not found` | 위와 같은 원인. docker 도 같이 설치가 취소된 상태입니다 |
| `could not select device driver "nvidia"` | 툴킷은 깔렸는데 런타임 미연결. `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `CondaToSNonInteractiveError` | 이 레포 스크립트는 conda-forge 를 쓰므로 안 납니다. 직접 `conda create` 를 쳤다면 `-c conda-forge --override-channels` 를 붙이세요 |
| `no space left on device` | 호스트에서 `docker system prune -af`. 100GB 는 필요합니다 |
| `ModuleNotFoundError: No module named 'pkg_resources'` (flatdict 빌드 중) | setuptools 82.0.0 이 pkg_resources 를 없앴고 pip 빌드 격리는 최신 setuptools 를 받습니다. `pip install --no-build-isolation flatdict==4.0.1` 먼저 |
| `No module named 'isaaclab'` (설치했는데도) | `isaaclab.sh -i` 는 확장 설치 실패를 종료코드에 반영하지 않습니다. `bash steps/diag.sh` 로 확인 후 `steps/40_isaaclab.sh` 재실행 |
| `Out of GPU memory` | `--num_envs` 를 절반으로 |
| `LLVM ERROR: out of memory` | VRAM 이 아니라 시스템 RAM. Windows 면 `.wslconfig` 의 swap 을 늘리세요 |
| `librcutils.so: cannot open shared object file` | `source /opt/isaaclab-scripts/isaac-env.sh` 다시 |
| 아무 반응 없이 멈춤 (첫 실행) | 셰이더 컴파일입니다. 5~10분 기다려 보세요 |

---

## 정리하고 싶을 때

```bash
# 컨테이너만
docker compose -f docker/isaaclab/docker-compose.yml down

# 설치 내용까지 전부 (10GB+ 다시 받아야 합니다)
docker compose -f docker/isaaclab/docker-compose.yml down -v
```

---

더 자세한 내용:
- [`README.md`](README.md) — 버전 조합, Humble/python 3.11 충돌 구조, 한 번에 빌드하는 경로
- [`steps/README.md`](steps/README.md) — 각 단계가 뭘 하는지, 바꿀 수 있는 값
- [`steps/rtx4060.env`](steps/rtx4060.env) — 8GB VRAM 설정값과 OOM 해석
