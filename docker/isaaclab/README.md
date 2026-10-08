# Isaac Sim 5.1 + Isaac Lab 2.3.1 도커 환경

> **터미널에 칠 명령만 순서대로 보려면 → [QUICKSTART.md](QUICKSTART.md)**
> 이 문서는 왜 그렇게 되어 있는지(버전 조합, Humble/python 3.11 충돌)를 설명합니다.

| 항목 | 버전 | 비고 |
|---|---|---|
| OS (컨테이너) | Ubuntu 22.04 (jammy) | GLIBC 2.35 — Isaac Sim pip 설치 하한선에 정확히 걸립니다 |
| CUDA (베이스 이미지) | 12.8.1 | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| Python | 3.11 | Isaac Sim 5.x 는 3.11 **고정**. 다른 버전은 에러 |
| PyTorch | 2.7.0 / torchvision 0.22.0 | `--index-url .../whl/cu128` |
| Isaac Sim | 5.1.0 | `isaacsim[all,extscache]==5.1.0` (pypi.nvidia.com) |
| Isaac Lab | v2.3.1 | 소스 클론 + `./isaaclab.sh -i` |
| ROS 2 | Humble | 시스템 python 3.10 쪽 |
| 가상환경 | miniconda `isaac_lab` | `/opt/conda/envs/isaac_lab` |

---

## 먼저 알아야 할 것: Humble ↔ Python 3.11 충돌

이 조합의 유일한 구조적 문제입니다. 미리 보고 가는 게 낫습니다.

- Ubuntu 22.04 의 ROS 2 Humble apt 패키지는 **시스템 python 3.10** 용으로 빌드돼 있습니다.
- Isaac Sim 5.1 은 **python 3.11 만** 지원합니다.
- 따라서 `conda activate isaac_lab` 한 상태에서 `/opt/ros/humble/setup.bash` 를 source 하고
  `import rclpy` 하면 **반드시 실패합니다.** 설정을 어떻게 만져도 안 됩니다.
  (apt 패키지를 3.11 로 쓰는 방법은 없고, rclpy 를 3.11 로 소스 빌드하는 길만 남습니다.)

그래서 이 이미지는 **쉘을 두 개로 쪼개** 놓았습니다.

| 쉘 | 파이썬 | 쓰는 곳 |
|---|---|---|
| `isaac-shell` | conda `isaac_lab` (3.11) | Isaac Sim, Isaac Lab 학습/추론 스크립트 |
| `ros-shell` | 시스템 (3.10) | `ros2` CLI, rviz2, Nav2, 직접 쓴 rclpy 노드 |

둘은 **DDS 로** 통신합니다. 데이터 전송은 DDS 가 하니까 양쪽 파이썬 버전이 달라도 상관없습니다.
Isaac Sim 쪽은 ROS 2 브리지가 품고 있는 **내부 Humble 라이브러리**를 쓰고
(`isaac-env.sh` 가 `LD_LIBRARY_PATH` 에 경로를 한 번만 추가),
ROS 노드 쪽은 시스템 Humble 을 그대로 씁니다.

> ⚠️ `isaac-shell` 안에서 `/opt/ros/humble/setup.bash` 를 source 하지 마세요.
> 3.10 심볼이 섞여 Kit 기동 자체가 깨집니다. 이건 NVIDIA 쪽에서도 공식적으로 경고하는 항목입니다.

커스텀 메시지가 필요하면 선택지가 둘입니다.
1. 메시지 패키지만 python 3.11 로 따로 빌드해서 Isaac 쪽 `LD_LIBRARY_PATH` 에 넣기
2. Isaac Sim 쪽은 표준 메시지만 쓰고, 변환 노드를 `ros-shell` 쪽에 두기 — 보통 이게 쌉니다

---

## 호스트 준비물

```bash
# 1. 드라이버 — Isaac Sim 5.1 테스트 버전은 580.65.06 입니다. 최소 CUDA 12.8 을 받으려면 570+ 필요
nvidia-smi

# 2. Docker + nvidia-container-toolkit
#    nvidia-container-toolkit 은 우분투 기본 저장소에 없습니다 (NVIDIA 저장소 등록 필요).
#    저장소 등록 → 설치 → 런타임 연결까지 한 번에:
bash docker/isaaclab/host_setup.sh
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi   # 여기까지 돼야 합니다

# 3. GUI 를 쓸 거면
xhost +local:root
```

GPU 는 **RT 코어가 있어야** 합니다. A100 / H100 은 RT 코어가 없어 Isaac Sim 이 지원하지 않습니다.
공식 최소 사양은 RTX 4080 16GB, RAM 32GB(권장 64GB), SSD 50GB 이상입니다.
실제로는 `isaacsim[all,extscache]` 만 10GB 가 넘으니 **디스크는 100GB 쯤 비워 두세요.**

---

## 두 가지 설치 경로

| | 언제 |
|---|---|
| **`Dockerfile`** (한 번에) | 데스크톱, 디스크·시간 여유 있을 때 |
| **[`steps/`](steps/README.md)** (끊어서) | **노트북 권장.** 단계별로 돌리고 깨진 지점부터 이어서 |

isaacsim 다운로드가 10GB 를 넘어서, 뒤에서 깨지면 전부 날아갑니다.
VRAM/RAM 이 빠듯한 노트북이면 `steps/00_preflight.sh` 로 점검부터 하세요.

```bash
bash docker/isaaclab/steps/00_preflight.sh   # 아무것도 안 깔고 점검만
bash docker/isaaclab/steps/10_conda_env.sh   # miniconda + isaac_lab (python 3.11)
# ... 20 → 30 → 40 → 50
```

## 빌드 & 실행

```bash
cd <레포 루트>

# 빌드 (30~90분. isaacsim 다운로드가 대부분입니다)
docker compose -f docker/isaaclab/docker-compose.yml build

# 쉘 진입 — 들어가면 이미 conda isaac_lab 이 활성화돼 있습니다
docker compose -f docker/isaaclab/docker-compose.yml run --rm isaaclab

# 설치 검증
docker compose -f docker/isaaclab/docker-compose.yml run --rm isaaclab \
  bash /workspace/rospider/docker/isaaclab/smoke_test.sh
```

ROS 2 패키지를 가볍게 가고 싶으면:

```bash
docker compose -f docker/isaaclab/docker-compose.yml build \
  --build-arg ROS_APT_PACKAGES="ros-humble-ros-base ros-humble-rmw-fastrtps-cpp"
```

### 컨테이너 안에서

```bash
# Isaac Lab 튜토리얼 (헤드리스)
isaaclab -p ${ISAACLAB_PATH}/scripts/tutorials/00_sim/create_empty.py --headless

# GUI
isaaclab -p ${ISAACLAB_PATH}/scripts/tutorials/00_sim/create_empty.py

# 학습 예시
isaaclab -p ${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --num_envs 4096

# ROS 2 쪽으로 갈아타기 (같은 컨테이너, 다른 쉘)
ros-shell
# 또는 한 줄만
ros-shell ros2 topic list
```

헤드리스 + 브라우저로 보려면 WebRTC 스트리밍(8211/udp, 49100/tcp)을 쓰면 됩니다.
`network_mode: host` 라 포트는 이미 열려 있습니다.

---

## 도커 없이 그냥 깔 때 (같은 순서)

```bash
conda create -n isaac_lab python=3.11
conda activate isaac_lab
pip install --upgrade pip

# Isaac Sim 5.1
ldd --version                       # 2.35 이상인지 확인
export OMNI_KIT_ACCEPT_EULA=YES
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# PyTorch — isaacsim 이 끌고 온 torch 를 덮어씁니다
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab 2.3.1
git clone --branch v2.3.1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh -i

# ★ 마지막에 torch 를 다시 확인하세요.
#   isaaclab.sh -i 가 rl_games / rsl_rl / sb3 / skrl 을 깔면서 torch 를 갈아치우는 경우가 있습니다.
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.7.0+cu128 12.8
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## 막힐 때

| 증상 | 원인 / 손볼 곳 |
|---|---|
| `librcutils.so: cannot open shared object file` | 내부 ROS 2 라이브러리 경로가 `LD_LIBRARY_PATH` 에 없습니다. `source /opt/isaaclab-scripts/isaac-env.sh` 다시 실행하고 `echo $ISAAC_ROS_LIB_PATH` 확인 |
| `GLIBCXX_3.4.30 not found` (RMW 에서) | conda 의 `libstdc++` 가 시스템 것을 가렸습니다. ROS 는 `ros-shell` 에서만 돌리세요 |
| `rclpy` import 실패 | 3.10/3.11 충돌. 위 "구조적 문제" 절 참고. `isaac-shell` 에서 rclpy 를 쓰려고 한 게 아닌지 확인 |
| Kit 가 기동 중에 심볼 충돌로 죽음 | 같은 쉘에서 ROS 를 source 했습니다. 새 쉘(`isaac-shell`)에서 다시 |
| `No device could be created` (gpu.foundation.plugin) | 드라이버 / `--gpus all` / RT 코어 없는 GPU |
| `LD_LIBRARY_PATH` 가 계속 길어짐 | `isaac-env.sh` 를 여러 번 source. `ISAAC_ROS_LIB_ADDED` 가드가 막지만, 수동으로 export 했다면 쉘을 새로 여세요 |
| 컨테이너 띄울 때마다 확장/셰이더 재다운로드 | compose 의 캐시 볼륨을 지웠는지 확인 (`isaac-cache-ov` 등) |
| torch 가 2.7.0 이 아님 | `./isaaclab.sh -i` 가 덮었습니다. 위 ★ 단계 재실행 |
| 빌드 중 디스크 부족 | `docker system prune -af` 후 재시도. 100GB 는 있어야 합니다 |

ROS 2 양쪽이 서로 보이는지 확인:

```bash
# 터미널 A (Isaac 쪽) — 시뮬 돌리면서 토픽 publish
# 터미널 B
docker compose -f docker/isaaclab/docker-compose.yml exec isaaclab ros-shell ros2 topic list
```

`ROS_DOMAIN_ID` 와 `RMW_IMPLEMENTATION` 이 양쪽에서 같아야 합니다 (이미지 기본값: `0`, `rmw_fastrtps_cpp`).

---

## 실기(RosPider)와 붙일 때

이 컨테이너는 **ROS 2 Humble** 입니다. 실기와 시뮬을 같이 쓸 거라면 먼저 확인하세요:

```bash
# RosPider 보드(라즈베리파이 등)에 ssh 로 들어가서
echo $ROS_DISTRO
```

- **실기도 Humble** 이면 그대로 맞습니다. `ROS_DOMAIN_ID` 와 `RMW_IMPLEMENTATION` 만
  양쪽에서 같게 두면 DDS 로 바로 붙습니다 (이 이미지 기본값: `0`, `rmw_fastrtps_cpp`).
- **다른 배포판**(Jazzy, Foxy, 또는 ROS 1)이면 **DDS 레벨 상호운용이 보장되지 않습니다.**
  둘 중 하나로 통일하는 게 안전합니다. 컨테이너 쪽을 바꾸려면:

  ```bash
  docker compose -f docker/isaaclab/docker-compose.yml build base \
    --build-arg ROS_DISTRO=jazzy \
    --build-arg ROS_APT_PACKAGES="ros-jazzy-ros-base ros-jazzy-rmw-fastrtps-cpp"
  ```

  단, **Jazzy 는 Ubuntu 24.04 용**이라 베이스 이미지(`CUDA_IMAGE`)도 24.04 로 같이
  바꿔야 합니다. 그러면 시스템 파이썬이 3.12 가 되는데, Isaac Sim 5.1 은 3.11 고정이라
  conda 쪽은 그대로 3.11 로 둡니다 — 쉘 분리 구조(`isaac-shell` / `ros-shell`)는 동일합니다.
  ROS 1 이라면 브리지(`ros1_bridge`)가 따로 필요합니다.

> 이 문단은 RosPider 실기 설정을 직접 확인하지 않고 쓴 것입니다.
> `echo $ROS_DISTRO` 결과를 보고 맞는 쪽으로 정리하세요.

---

## 참고 문서

- [Isaac Lab v2.3.1 — Installation using Isaac Sim Pip Package](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html)
- [Isaac Lab — Local Installation (개요)](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Sim 5.1 — Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Isaac Sim 5.1 — Python Environment Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Sim 5.1 — ROS 2 Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [NVIDIA 포럼 — rclpy import failure, Python/Humble 버전 충돌](https://forums.developer.nvidia.com/t/rclpy-import-failure-in-isaac-sim-due-to-python-version-conflict-with-ros2-humble-seeking-robust-docker-solution/351672)
