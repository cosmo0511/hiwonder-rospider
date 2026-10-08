# Hiwonder RosPider

시뮬레이션 환경부터 올려두는 중입니다.

## Isaac Sim 5.1 + Isaac Lab 2.3.1

| | |
|---|---|
| **[docker/isaaclab/QUICKSTART.md](docker/isaaclab/QUICKSTART.md)** | **터미널에 칠 명령 순서대로.** 여기서 시작하세요 |
| [docker/isaaclab/README.md](docker/isaaclab/README.md) | 버전 조합, ROS 2 / python 3.11 충돌 구조, 트러블슈팅 |
| [docker/isaaclab/steps/README.md](docker/isaaclab/steps/README.md) | 단계별 스크립트가 각각 뭘 하는지 |
| [docker/isaaclab/steps/rtx4060.env](docker/isaaclab/steps/rtx4060.env) | VRAM 8GB 급 설정값, OOM 메시지 해석 |

```bash
bash docker/isaaclab/host_setup.sh                                  # 호스트 1회
docker compose -f docker/isaaclab/docker-compose.yml build base
docker compose -f docker/isaaclab/docker-compose.yml run --rm base
# 컨테이너 안에서
bash /opt/isaaclab-steps/run_all.sh
```

### 고정한 버전

| 항목 | 버전 | 비고 |
|---|---|---|
| OS (컨테이너) | Ubuntu 22.04 | GLIBC 2.35 — Isaac Sim pip 설치 하한선 |
| CUDA (베이스 이미지) | 12.8.1 | 드라이버 570+ 필요 (5.1 테스트 버전 580.65.06) |
| Python | 3.11 | Isaac Sim 5.x **고정**. 다른 버전은 동작하지 않음 |
| PyTorch | 2.7.0 / torchvision 0.22.0 | cu128 |
| Isaac Sim | 5.1.0 | `isaacsim[all,extscache]` |
| Isaac Lab | v2.3.1 | 소스 클론 + `./isaaclab.sh -i` |
| ROS 2 | Humble | 시스템 python 3.10 쪽 |
| 가상환경 | miniconda `isaac_lab` | |

### 알아둘 것 두 가지

**1. ROS 2 Humble 과 python 3.11 은 같은 쉘에서 못 씁니다.**
Humble 의 apt `rclpy` 는 python 3.10 용이고 Isaac Sim 5.1 은 3.11 고정입니다.
그래서 쉘을 둘로 나눠 DDS 로 붙입니다 — `isaac-shell`(3.11, 시뮬) / `ros-shell`(3.10, ROS 노드).
자세한 건 [docker/isaaclab/README.md](docker/isaaclab/README.md).

**2. RTX 4060 (8GB) 는 공식 최소(RTX 4080 16GB) 아래입니다.**
돌아가지만 학습은 `--headless` 필수, `--num_envs` 는 64 부터 올려가며 한계를 찾으세요.

## 센서 융합 과제 (뎁스 카메라 + IMU)

| | |
|---|---|
| **[docs/sensor_fusion_manual.md](docs/sensor_fusion_manual.md)** | **매뉴얼.** 시나리오, 좌표계·단위, 융합 수식, 실험 조건, 함정, 노션 노트 작성 항목까지 |
| [tasks/sensor_fusion/README.md](tasks/sensor_fusion/README.md) | 커맨드 모음 |

```bash
# 컨테이너 안에서 (매번 이 두 줄 -> 프롬프트에 (isaac_lab) 이 붙습니다)
source /opt/isaaclab-scripts/isaac-env.sh
cd /workspace/rospider

pip install xacro matplotlib                        # 최초 1회
bash tasks/sensor_fusion/00_fetch_description.sh    # RosPider URDF 받기
python tasks/sensor_fusion/01_xacro_to_urdf.py      # xacro -> URDF
python tasks/sensor_fusion/test_fusion.py           # 융합 로직만 먼저 검증 (Kit 불필요)
python tasks/sensor_fusion/02_urdf_to_usd.py        # URDF -> USD (--view 로 눈 확인)
bash tasks/sensor_fusion/run_all_conditions.sh      # 실험 조건 4개
```

Kit 이 필요 없는 단계를 앞에 몰아놨습니다. 융합 로직은 `AppLauncher` 없이 테스트되므로
문제가 생겼을 때 시뮬레이터 탓인지 로직 탓인지 바로 갈립니다.

## 검증 상태

RTX 4060 노트북 + Ubuntu 24.04 호스트에서 설치를 끝까지 돌리며 다듬었습니다.
`00_preflight` ~ `40_isaaclab` 통과, Kit 기동(GPU 인식, Vulkan)까지 확인했습니다.
`50_verify` 의 유한 스텝 검증은 아직 완주 기록이 없습니다.

센서 융합 쪽은 Kit 없이 도는 부분(xacro 전개, 외부 파라미터 계산, 융합 로직 7개 조건)만
실행 검증했습니다. Isaac Sim 안에서 끝까지 돌려본 기록은 아직 없습니다 —
자세한 구분은 [매뉴얼 0절](docs/sensor_fusion_manual.md#0-먼저-무엇이-확인된-사실인가).
