# CLAUDE.md

이 레포에서 작업하는 AI 에이전트를 위한 문서입니다. 사람도 읽으면 됩니다.

## 프로젝트

**Hiwonder RosPider** (6족 보행 로봇)를 Isaac Sim / Isaac Lab 에서 시뮬레이션합니다.

목표를 순서대로:

1. RosPider **URDF 를 올려** USD 로 변환하고 씬에 세우기
2. **뎁스 카메라 + IMU** 두 센서를 붙이고, 그 정보를 **결합하는 시뮬레이션** 구현

지금은 **2번 환경 구축만 끝난 상태**입니다. 로봇 에셋(URDF/USD/메시)도, 태스크 코드도
아직 레포에 없습니다. `docker/isaaclab/` 이 전부입니다.

## 하드웨어 (이 제약이 설계를 많이 좌우합니다)

| | |
|---|---|
| GPU | **RTX 4060 Laptop, VRAM 8188 MiB** |
| CPU / RAM | Ryzen 7 8845HS (8C/16T) / **15227 MiB** + 스왑 28671 MiB |
| 드라이버 | 580.173.02 |
| 호스트 OS | Ubuntu 24.04 (컨테이너 안은 22.04) |

**공식 최소 사양(RTX 4080 16GB, RAM 32GB) 아래입니다.** 돌아가지만:

- 학습·배치 실행은 `--headless` **필수**. GUI 렌더러가 VRAM 3~4GB 를 먼저 먹습니다
- `--num_envs` 기본값 4096 은 무조건 OOM. **64 에서 시작**해 두 배씩 올리며 한계를 찾으세요
- **카메라를 켜면 VRAM 소모가 급증합니다.** 뎁스 카메라를 붙인 뒤에는 num_envs 를
  다시 바닥부터 잡아야 합니다. 많은 env 가 필요하면 `Camera` 대신 `TiledCamera` 를 쓰세요
- `LLVM ERROR: out of memory` 는 VRAM 이 아니라 **시스템 RAM** 문제입니다
- 자세한 건 `docker/isaaclab/steps/rtx4060.env`

## 버전 핀 — 바꾸지 마세요

| 항목 | 버전 | 왜 고정인가 |
|---|---|---|
| Ubuntu (컨테이너) | 22.04 | GLIBC 2.35. Isaac Sim pip 설치 하한선에 정확히 걸침. 20.04(2.31)는 불가 |
| Python | **3.11** | Isaac Sim 5.x 고정. 3.10/3.12 는 동작하지 않음 |
| Isaac Sim | 5.1.0 | `isaacsim[all,extscache]==5.1.0`, `--extra-index-url https://pypi.nvidia.com` |
| Isaac Lab | v2.3.1 | 소스 클론 + `./isaaclab.sh -i` |
| PyTorch | 2.7.0 / torchvision 0.22.0 | `--index-url https://download.pytorch.org/whl/cu128` |
| CUDA (베이스 이미지) | 12.8.1 | 드라이버 570+ 필요 |
| ROS 2 | Humble | 시스템 python 3.10 쪽 |
| conda 환경 | `isaac_lab` | `/opt/conda/envs/isaac_lab` |

버전을 올리고 싶으면 `docker/isaaclab/steps/common.sh` 의 기본값을 환경변수로 덮어쓰면
됩니다. 다만 **python 3.11 은 Isaac Sim 쪽 제약이라 협상 불가**입니다.

## 구조적 제약: ROS 2 Humble 과 Python 3.11 은 같은 쉘에서 못 씁니다

22.04 의 Humble apt 패키지는 **python 3.10** 용으로 빌드돼 있고, Isaac Sim 5.1 은
**3.11 전용**입니다. `conda activate isaac_lab` 상태에서 `/opt/ros/humble/setup.bash` 를
source 하고 `import rclpy` 하면 **반드시 실패합니다.** 설정으로 우회 못 합니다.

그래서 쉘을 둘로 나눕니다:

| 쉘 | 파이썬 | 쓰는 곳 |
|---|---|---|
| `isaac-shell` | conda `isaac_lab` (3.11) | Isaac Sim, Isaac Lab 스크립트 |
| `ros-shell` | 시스템 (3.10) | `ros2` CLI, rviz2, 직접 쓴 rclpy 노드 |

둘은 **DDS** 로 통신합니다(`ROS_DOMAIN_ID=0`, `rmw_fastrtps_cpp`). Isaac Sim 쪽은
ros2 브리지가 품은 **내부 Humble 라이브러리**를 쓰고, `isaac-env.sh` 가 그 경로를
`LD_LIBRARY_PATH` 에 한 번만 넣습니다.

> **`isaac-shell` 안에서 절대 ROS 를 source 하지 마세요.** 3.10 심볼이 섞여 Kit 기동이 깨집니다.

커스텀 메시지가 필요하면: 메시지 패키지만 3.11 로 따로 빌드하거나, Isaac 쪽은 표준
메시지만 쓰고 변환 노드를 `ros-shell` 에 두세요. 보통 후자가 쌉니다.

## 다시 밟기 쉬운 함정

전부 **실기에서 한 번씩 터졌던 것들**입니다. 같은 증상을 보면 여기부터 보세요.

| 증상 | 원인과 해법 |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` (flatdict 빌드 중) | `flatdict==4.0.1` 은 sdist 뿐이라 소스 빌드 필수인데 setup.py 가 `pkg_resources` 를 import. 그건 **setuptools 82.0.0 에서 제거**됨. → `pip install "setuptools<82"` 후 `pip install --no-build-isolation flatdict==4.0.1`. **`PIP_CONSTRAINT` 는 빌드 격리 overlay 에 적용되지 않습니다**(pip 26 확인) |
| `No module named 'omni'` | **고장 아님.** `omni.*` 는 Kit 앱이 기동하며 주입됩니다. `isaaclab_tasks` / `isaaclab.envs` / `isaaclab.sim` 은 `AppLauncher` 로 앱을 띄운 **뒤에만** import 가능 |
| `No module named 'isaaclab'` (설치했는데도) | `isaaclab.sh -i` 는 확장 설치를 `find -exec bash -c` 서브셸로 돌려 **실패를 종료코드에 반영하지 않습니다.** 성공처럼 끝나도 확인이 필요 → `bash /opt/isaaclab-steps/diag.sh` |
| `torchaudio ... not installed` 경고 | `isaaclab.sh` 의 `ensure_cuda_torch()` 가 매번 `pip uninstall -y torch torchvision torchaudio` 후 torch/torchvision 만 재설치. torchaudio 를 안 되살림. `40_isaaclab.sh` 가 복구함 |
| `stable-baselines3 ... requires torch>=2.8` | sb3 최신판과 Isaac Lab 2.3.1 의 torch 2.7.0 핀은 **양립 불가**(상위 문제). sb3 안 쓰면 무시. 빼려면 `ISAACLAB_RL_FRAMEWORK=rsl_rl` |
| `ERROR: File or directory already exists: '/opt/conda'` | `/opt/conda` 는 **named volume 마운트 지점**이라 비어도 디렉터리는 존재. miniconda 설치기가 거부 → `-u` 플래그 |
| `CondaToSNonInteractiveError` | conda 24.x+ 가 Anaconda 기본 채널 ToS 동의를 요구. 이 레포는 **conda-forge 단독**(`-c conda-forge --override-channels`)이라 안 남 |
| 컨테이너가 바로 종료 | `set -e` 에서 **실패하는 명령치환 대입**은 그 자리에서 쉘을 죽입니다. 환경 준비 스크립트에는 `set -e` 를 쓰지 말 것 |
| 스크립트가 안 끝남 | 튜토리얼 `create_empty.py` 는 `while simulation_app.is_running(): sim.step()` **무한 루프**입니다. 검증에는 `steps/_smoke_sim.py`(유한 스텝) 사용 |
| `Authorization required, but no authorization protocol specified` | X11 권한. 호스트에서 `xhost +local:root` |

## 다음 작업에 쓸 API (v2.3.1 소스에서 확인함)

### URDF → USD

```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
```

### 센서

```python
from isaaclab.sensors import (
    Camera, CameraCfg, CameraData,          # 단일 카메라
    TiledCamera, TiledCameraCfg,            # 다수 env 용 (VRAM 효율적)
    Imu, ImuCfg, ImuData,
    ContactSensor, RayCaster, FrameTransformer,
)
```

**뎁스 카메라** — `CameraCfg.data_types` 기본값은 `["rgb"]` 입니다. 뎁스를 쓰려면 명시해야 합니다:

| 이름 | 뜻 |
|---|---|
| `"distance_to_camera"` | 카메라 광학 중심까지의 거리 |
| `"distance_to_image_plane"` | 카메라 z축 기준 이미지 평면까지의 거리 |
| `"depth"` | `"distance_to_image_plane"` 의 별칭 |

> **카메라 센서가 있으면 `--enable_cameras` 를 반드시 켜야 합니다.** 헤드리스에서도
> 필요합니다 (AppLauncher 문서: "This flag must be set to True if the environments
> contains any camera sensors").

**IMU** — `ImuCfg` 는 `offset`(pos/rot)과 `gravity_bias`(기본 `(0, 0, 9.81)`)를 받습니다.
`ImuData` 가 제공하는 필드:

```
pos_w, quat_w, projected_gravity_b, lin_vel_b, ang_vel_b, lin_acc_b, ang_acc_b
```

센서 융합(2번 목표)은 이 IMU 텐서와 뎁스 이미지를 observation 으로 묶는 형태가 됩니다.

### import 순서 (중요)

```python
from isaaclab.app import AppLauncher      # 기동 전엔 이것만 가능
app_launcher = AppLauncher(args_cli)      # ← Kit 기동
simulation_app = app_launcher.app
from isaaclab.sensors import CameraCfg    # ← 이제부터 omni 의존 모듈 import 가능
```

## 작업 방법

```bash
# 호스트 1회
bash docker/isaaclab/host_setup.sh
xhost +local:root                       # GUI 를 쓸 거면 필수

# 컨테이너 빌드 & 진입
docker compose -f docker/isaaclab/docker-compose.yml build base
docker compose -f docker/isaaclab/docker-compose.yml run --rm base

# 컨테이너 안 — 설치(이미 돼 있으면 건너뜀)
bash /opt/isaaclab-steps/run_all.sh
bash /opt/isaaclab-steps/diag.sh        # 상태 점검, 아무것도 안 바꿈
bash /opt/isaaclab-steps/50_verify.sh   # 검증 (헤드리스)
GUI=1 bash /opt/isaaclab-steps/50_verify.sh   # 창을 띄워서 눈으로 확인
```

GUI 는 기본이 꺼져 있습니다. 검증은 사람 없이 끝나야 하고, VRAM 8GB 에서는
GUI 렌더러가 3~4GB 를 먼저 먹기 때문입니다. **다만 URDF 를 올리고 센서를 붙이는
작업은 눈으로 봐야 하므로 GUI 를 쓰세요.** 창이 안 뜨고
`Authorization required, but no authorization protocol specified` 가 보이면
호스트에서 `xhost +local:root` 를 실행하지 않은 것입니다.

- 레포는 컨테이너 안 **`/workspace/rospider`** 에 마운트됩니다
- 설치는 docker named volume(`isaaclab_isaac-conda`, `isaaclab_isaac-lab`)에 있어
  컨테이너를 `--rm` 으로 지워도 남습니다. **compose 프로젝트명이 `isaaclab`**(파일이 있는
  디렉터리 이름)이라 경로 `docker/isaaclab/` 을 유지하는 한 볼륨이 재사용됩니다
- 설치 로그: `/opt/.isaac-setup-state/isaaclab_install.log`, `smoke_sim.log`
- 단계 완료 마커: `/opt/.isaac-setup-state/`. 다시 돌리려면 `FORCE=1`

스크립트를 고칠 때는 `/workspace/rospider/docker/isaaclab/steps/` 쪽을 쓰면 이미지
재빌드 없이 바로 반영됩니다. `/opt/isaaclab-steps/` 는 이미지에 구워진 사본입니다.

## 아직 검증 안 된 것 (사실로 쓰지 마세요)

- **`50_verify.sh` 완주 기록 없음.** Kit 기동(GPU 인식, Vulkan)까지는 확인했지만
  `[SMOKE-OK]` 를 본 적이 없습니다
- **RosPider 실기의 ROS 배포판 미확인.** 컨테이너는 Humble 입니다. 실기가 다르면
  DDS 상호운용이 보장되지 않습니다. 보드에서 `echo $ROS_DISTRO` 로 확인 필요
- **Dockerfile 의 full 스테이지(한 번에 빌드)는 끝까지 돌려본 적이 없습니다.**
  단계별 경로(`steps/`)만 실기 검증됐습니다
- **8GB VRAM 에서 실제로 되는 `num_envs` 상한 미측정.** 카메라 없이 64 부터 시작하세요

## 글쓰기 규칙

문서와 주석은 **한국어**로 씁니다. 커밋 메시지도 한국어로, 증상 → 원인 → 해법 순서로
적습니다. 추측과 확인된 사실을 섞지 말고, 확인 안 한 것은 그렇다고 표시하세요.
