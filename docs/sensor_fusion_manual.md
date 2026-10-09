# 센서 융합 과제 매뉴얼 — RosPider 뎁스 카메라 + IMU

> 과제: *NVIDIA Isaac Sim 기반 센서 융합으로 주변 환경 인식하기*
> 조합: **Depth + IMU** / 플랫폼: **Hiwonder RosPider** (6족) / Isaac Sim 5.1.0 + Isaac Lab 2.3.1

이 문서는 "어떤 코드를 써야 하는가" 에 대한 답이다. 실행 순서, 각 파일이 왜 그렇게
생겼는지, 어디서 터지는지, 노션 노트의 어느 칸을 무엇으로 채우는지까지 적었다.

## 0. 먼저, 무엇이 확인된 사실인가

이 레포의 규칙대로 추측과 확인을 섞지 않는다.

**확인한 것** (이 매뉴얼을 쓰면서 실제로 돌려봤다)

- RosPider URDF 는 실재한다. `github.com/Hiwonder/ROSpider` 의
  `src/simulations/rospider_description/` 에 xacro 와 STL 36개가 들어 있다.
- 그 xacro 는 ROS 없이 전개된다. `pip install xacro` + 경로 치환으로
  **링크 51개 / 조인트 50개 (revolute 29개: 다리 18 + 팔 5 + 그리퍼 6)** 의
  단일 URDF 가 나오고, 참조된 메시 파일이 전부 존재한다.
- 센서 장착 위치와 `R_cam<-imu` 는 URDF 체인에서 직접 계산한 값이다
  (`urdf_fk.py`). 손으로 적은 추정치가 아니다.
- 융합 로직(`fusion.py`) 은 합성 뎁스로 **7개 조건 전부 통과**한다
  (`test_fusion.py`). 6절의 숫자 표가 그 실행 결과다.
- 아래 Isaac Lab API 이름·필드는 v2.3.1 소스에서 확인했다
  (`UrdfConverterCfg`, `CameraCfg`, `ImuCfg`, `ImuData`, `CameraData`,
  `PinholeCameraCfg`, `ArticulationCfg`, `RigidBodyPropertiesCfg.kinematic_enabled`).

**아직 확인 안 한 것** (사실로 쓰지 말 것)

- **Isaac Sim 안에서 끝까지 돌려본 기록이 없다.** `02_urdf_to_usd.py` 와
  `run_fusion_demo.py` 는 검증된 API 로 작성했지만 Kit 을 띄워 완주시킨 적은 없다.
  이 레포는 `50_verify.sh` 완주 기록(`[SMOKE-OK]`)조차 아직 없는 상태다.
- **서 있는 다리 자세 각도.** `rospider_cfg.py` 의 다리 조인트 목표값은 0 이다.
  세 조인트(coxa/femur/tibla) 의 부호 규약을 GUI 로 보면서 맞춰야 한다.
- **PD 게인.** 출발점만 적어 뒀다.
- **8 GB VRAM 에서 카메라를 켠 채 돌아가는 상한.** `num_envs=1`, 160x120 으로
  시작하게 해 뒀다.

## 1. 요구사항 ↔ 산출물 매핑

| 과제가 요구하는 것 | 어디서 나오나 |
|---|---|
| 환경 구성 (바닥·장애물·관측 대상) | `rospider_cfg.py` 의 `build_scene_cfg()` |
| 센서 2종 선택과 이유 | 2절, 그리고 `fusion.py` 모듈 docstring |
| 데이터 수집 | `run_fusion_demo.py` 의 루프 (`camera.data.output`, `imu.data`) |
| 융합 → 하나의 결과 | `fusion.py` 의 `fuse()` → `decision` 한 개 |
| 단위·좌표계·측정 시점 | 4절 표 |
| 조건 2가지 이상 비교 | 6절. `--pitch_deg`, `--obstacle_height`, `--dists` |
| 단일 센서와의 차이 | `d_naive`, `d_roi` 를 같은 스텝에 같이 기록 |
| 실행 가능한 코드 | `tasks/sensor_fusion/` 전체 |

## 2. 시나리오: 왜 이 두 센서인가

**상황.** RosPider 가 평평하지 않은 바닥을 걷는다. 전방에 물체가 있으면 멈춰야 하고,
넘어갈 수 있는 낮은 단차면 그냥 가야 한다.

**뎁스 카메라만 쓰면 왜 안 되나.** RosPider 의 뎁스 카메라는 팔 끝에 달려 있어
전방을 **24.2도 아래로** 본다. 그래서 화면 아래쪽은 언제나 바닥이다. "최소 뎁스" 로
장애물을 찾으면 **바닥이 1등으로 잡힌다.** 흔한 대응은 화면 위쪽만 잘라 쓰는 것인데
(= 고정 ROI), 6족 보행은 몸체가 수시로 끄덕인다. 몸이 20도 숙으면 잘라낸 영역 안으로
바닥이 들어와 **오경보**가 난다.

**IMU 가 메우는 구멍.** IMU 는 거리를 못 재지만 **중력 방향**을 안다. 중력을 알면
"이 점이 바닥에서 몇 cm 높은가" 를 계산할 수 있고, 그러면 바닥을 **기하학적으로**
제거할 수 있다. 화면 어디를 보는지와 무관해진다.

**선택 이유 한 줄 요약**

| 센서 | 역할 | 왜 필요한가 |
|---|---|---|
| Depth (`distance_to_image_plane`) | 전방 물체까지의 **거리**와 모양 | 거리 없이는 "얼마나 가까운가" 를 못 판단한다 |
| IMU (`projected_gravity_b`, `lin_acc_b`) | 몸체 **자세**와 **충격** | 자세 없이는 뎁스의 어느 점이 바닥인지 모른다 |

이게 과제가 요구하는 "센서별 화면을 나란히 놓는 것에서 더 나아가, 두 센서가 최종
판단에 어떻게 함께 쓰이는지" 에 대한 답이다. 두 센서가 **하나의 판정 변수**로 합쳐진다.

## 3. 파이프라인

```
 ┌─ 뎁스 (N,H,W) [m] ──► 역투영 ──► 점 집합 p (카메라 광학 프레임)
 │        intrinsics K          x=(u-cx)/fx·z, y=(v-cy)/fy·z, z=depth
 │                                              │
 │                                              ▼
 │                         h = p·up − (바닥면)   ◄── up = −R_cam←imu · g_imu
 │                         d = p·forward                     ▲
 │                         l = p·right                       │
 │                                              │            │
 └─ IMU projected_gravity_b (N,3) ──────────────┴────────────┘
                lin_acc_b (N,3) ──┐
                                   │
   마스크: 0.03 < h < 0.40 그리고 |l| < 0.15 그리고 유효거리
                                   │
   d_fused = min(d | 마스크)        │
                                   ▼
   판정  tilt>25° → STOP_TILT │ |a|>6 → STOP_IMPACT
         d_fused<0.30 → DANGER │ <0.60 → WARN │ else CLEAR
```

핵심은 **IMU 가 뎁스 점의 좌표계를 중력에 정렬시키고, 정렬된 좌표에서만 거리 판정이
일어난다**는 것이다. 두 센서 중 하나를 빼면 파이프라인이 끊긴다.

## 4. 단위·좌표계·측정 시점 (노션 노트에 그대로 쓸 표)

| 항목 | 값 | 비고 |
|---|---|---|
| 뎁스 타입 | `distance_to_image_plane` | `"depth"` 는 이것의 별칭. `CameraCfg.data_types` 기본값은 `["rgb"]` 라 **명시 필수** |
| 뎁스 단위 | m, 카메라 광학 z축 기준 | 광심까지의 거리를 원하면 `distance_to_camera` |
| 측정 못 한 픽셀 | `inf` | `depth_clipping_behavior="none"` (기본). `isfinite` 로 걸러야 한다 |
| 카메라 프레임 | ROS 광학: **+X 오른쪽, +Y 아래, +Z 시선** | `CameraCfg.OffsetCfg(convention=...)` 가 `"ros"`/`"opengl"`/`"world"` 중 어떤 규약으로 오프셋을 해석할지 정한다 |
| 해상도 / 화각 | 160x120 / 수평 70도 | `focal_length=14.96`, `horizontal_aperture=20.955` → `2·atan(20.955/(2·14.96))=70°` |
| 내부 행렬 | fx=fy=114.25 px, cx=80, cy=60 | 손으로 계산하지 말고 `camera.data.intrinsic_matrices` 를 쓰면 된다 |
| 카메라 위치 | base_link 기준 (0.13125, 0.0013, 0.25086) m | URDF 체인에서 계산 (`urdf_fk.py`) |
| 카메라 방향 | 전방 기준 **24.17도 하향** | world 규약 쿼터니언 (0.97753, 0, 0.210794, 0) |
| 카메라 높이 | 수평 자세에서 바닥 위 **0.367 m** | base_link 0.11607 + 0.25086 |
| IMU 위치 | base_link 기준 (0.0048416, 0.011168, −0.0057398) m | URDF `imu_joint` 그대로 |
| IMU 방향 | base_link 대비 **yaw −90도** | 쿼터니언 (0.70711, 0, 0, −0.70711) |
| 중력 벡터 | 크기 1 단위벡터, 아래 방향 | 수평일 때 `projected_gravity_b = (0,0,−1)` |
| 가속도 | m/s², IMU 프레임 | `gravity_bias=(0,0,0)` 으로 두면 **순수 운동가속**. 기본값 (0,0,9.81) 은 실제 IMU 처럼 정지 시 +9.81 |
| 각도 | 내부 rad, 출력 deg | |
| 측정 시점 | **두 센서 모두 `update_period=0.0`** | 같은 물리 스텝에서 갱신되므로 시점 차이가 없다. 카메라만 주기를 키우면 뎁스가 묵은 값이 되어, 기울기가 빠르게 변할 때 융합이 밀린다 (7절 '한계') |

### 이 장착이 만드는 사각지대 (반드시 노트에 적을 것)

카메라 높이 0.367 m, 하향 24.17도, 수직 반화각 `atan(60/114.25)=27.71도` 이므로:

- 화면 **맨 아래** 광선은 수평 아래 **51.88도** → 바닥을 `0.367/tan(51.88°) = 0.288 m` 에서 맞춘다
- 화면 **중앙** 광선은 24.17도 → 바닥을 **0.817 m** 에서 맞춘다
- 즉 **카메라 앞 0.29 m 안쪽 바닥은 안 보인다.** 높이 0.15 m 물체는 0.17 m 부터 보인다

그래서 경고 임계값을 `DANGER < 0.30 m`, `WARN < 0.60 m` 로 잡았다. 0.15 m 같은 값을
쓰면 **사각지대 안이라 애초에 측정이 안 되는 거리**를 임계값으로 쓰게 된다. 임계값은
센서 기하에서 역산해야 한다.

## 5. 어떻게 돌리나

### 5.0 환경이 셋이다 — 무엇이 어디서 도나

가장 헷갈리는 지점이라 먼저 못박는다. **코드를 쓰는 곳과 시뮬레이션이 도는 곳이 다르다.**

| | ① 코드 작성 환경 | ② 노트북 (호스트) | ③ 도커 컨테이너 |
|---|---|---|---|
| 무엇 | 에디터 / Claude Code 클라우드 세션 | 실제 하드웨어 | **시뮬레이션이 도는 곳** |
| OS | (상관없음) | **Ubuntu 24.04** | **Ubuntu 22.04** |
| Python | (상관없음) | (상관없음) | **3.11** (conda `isaac_lab`) |
| GPU | 없어도 된다 | RTX 4060 Laptop **8188 MiB** | 호스트 GPU 를 그대로 쓴다 |
| Isaac Sim | 없다 | 없다 | **5.1.0** |
| 여기서 하는 일 | 코드 작성, `git push` | `docker` / `xhost` / `nvidia-smi` **뿐** | **모든 python 명령** |
| 레포 경로 | 각자 클론한 곳 | `~/hiwonder-rospider` | `/workspace/rospider` |

```
  ① 코드 작성 (GPU 불필요)
       └─ git push ──► GitHub ──► git pull ──┐
                                              ▼
                        ② 노트북 / 호스트 (Ubuntu 24.04 + RTX 4060 + 드라이버 580)
                              └─ docker compose run --rm base
                                    └─ ③ 컨테이너 (Ubuntu 22.04 + py3.11 + Isaac Sim 5.1)
                                          └─ /workspace/rospider 에서 실행
```

**호스트가 24.04 인데 컨테이너가 22.04 인 것은 맞다.** 컨테이너가 자기 userspace 를
들고 오고, 호스트에서 공유되는 것은 NVIDIA 드라이버뿐이다. 그래서 호스트 OS 버전은
거의 상관없고, **드라이버 버전(570+, 이 레포는 580.173.02)만 중요하다.**
컨테이너를 22.04 로 고정한 이유는 GLIBC 2.35 가 Isaac Sim pip 설치의 하한선이고
ROS 2 Humble 이 22.04 용으로 빌드돼 있기 때문이다. 자세한 건 레포 루트 `CLAUDE.md`.

이 과제의 **0~3단계(5.4절) 는 GPU 도 Isaac Sim 도 필요 없다.** xacro 전개, 기구학
계산, 융합 로직 테스트는 ① 에서도 돌아간다. 4단계부터가 ③ 전용이다.

### 5.0.1 명령을 어디서 치는지 — 프롬프트로 구분

| 프롬프트 | 어디 | 무엇을 치나 |
|---|---|---|
| `user@노트북:~/hiwonder-rospider$` | **② 호스트** | docker 명령, `xhost`, `nvidia-smi` |
| `root@...:/workspace#` | **③ 컨테이너** (conda 비활성) | `source .../isaac-env.sh` 한 줄 |
| `(isaac_lab) root@...:/workspace/rospider#` | **③ 컨테이너 + conda 활성** | 이 과제의 모든 python 명령 |

레포는 컨테이너 안 **`/workspace/rospider`** 에 bind mount 된다. 호스트에서 수정한 파일이
즉시 보이고, 컨테이너가 만든 결과물(`outputs/`)도 호스트 레포 안에 그대로 남는다.
**결과를 꺼내려고 `docker cp` 를 할 필요가 없다.**

### 5.0.2 전체 사양 한눈에

노션 노트의 "실행 환경" 칸에 그대로 옮길 표다.

| 항목 | 값 | 비고 |
|---|---|---|
| 호스트 OS | Ubuntu 24.04 | 컨테이너가 자기 userspace 를 들고 오므로 버전 제약 거의 없음 |
| GPU | NVIDIA RTX 4060 Laptop, VRAM 8188 MiB | **공식 최소(RTX 4080 16GB) 아래.** 돌아가지만 설정을 줄여야 한다 |
| 드라이버 | 580.173.02 | CUDA 12.8 에 570+ 필요 |
| CPU / RAM | Ryzen 7 8845HS (8C/16T) / 15227 MiB + 스왑 28671 MiB | |
| 컨테이너 OS | Ubuntu 22.04 | GLIBC 2.35 = Isaac Sim pip 하한선 |
| Python | 3.11 (conda `isaac_lab`) | Isaac Sim 5.x 고정. **협상 불가** |
| Isaac Sim | 5.1.0 | `isaacsim[all,extscache]==5.1.0` |
| Isaac Lab | v2.3.1 | 소스 클론 + `./isaaclab.sh -i` |
| PyTorch | 2.7.0 / torchvision 0.22.0 (cu128) | |
| CUDA (베이스 이미지) | 12.8.1 | |
| ROS 2 | Humble | python 3.10 쪽. **이 과제는 쓰지 않는다** |
| 추가 설치 | `xacro`, `matplotlib` | `pip install` (5.3절) |
| 이 과제 설정 | `num_envs=1`, 카메라 160x120, 수평화각 70도 | 8 GB VRAM 기준 |

실기 로봇(RosPider 보드) 은 Jetson Orin Nano + Ubuntu 22.04 + ROS 2 Humble 로 알려져
있으나 **이 레포에서 확인한 바 없다.** 보드에서 `echo $ROS_DISTRO` 로 확인해야 한다.
이 과제는 실기와 통신하지 않으므로 지금은 영향이 없다.

### 5.1 호스트: 컨테이너 띄우기

Isaac Sim / Isaac Lab 설치가 아직이면 먼저
[`docker/isaaclab/QUICKSTART.md`](../docker/isaaclab/QUICKSTART.md) 를 끝내고 오자.
여기서는 설치가 끝난 상태를 가정한다.

```bash
# 이 과제 코드가 올라간 브랜치를 받는다 (처음 한 번)
cd ~/hiwonder-rospider           # 레포를 클론한 곳. 없으면
                                 #   git clone https://github.com/cosmo0511/hiwonder-rospider.git
git fetch origin
git checkout claude/great-tesla-700vl0
git pull

xhost +local:root                # GUI 를 쓸 거면 매 로그인마다 1회
docker compose -f docker/isaaclab/docker-compose.yml run --rm base
```

> **`base` 서비스를 쓴다.** 설치가 담긴 도커 볼륨(`isaac-conda`, `isaac-lab`)이
> `base` 에만 붙어 있다. `isaaclab`(full) 서비스는 한 번에 빌드하는 경로이고
> 이 레포에서 완주 검증이 안 됐다.

프롬프트가 `root@...:/workspace#` 로 바뀌면 들어온 것이다.

### 5.2 컨테이너: conda 활성화 (매번 1회)

```bash
source /opt/isaaclab-scripts/isaac-env.sh
cd /workspace/rospider
```

프롬프트에 **`(isaac_lab)`** 이 붙으면 된 것이다. 다만 **안 붙어도 정상일 수 있다**
(아래 참고). 프롬프트는 믿지 말고 이걸로 확인하자:

```bash
which python && python --version && echo "ENV=$CONDA_DEFAULT_ENV"
```

```
/opt/conda/envs/isaac_lab/bin/python
Python 3.11.x
ENV=isaac_lab                       <- 이러면 된 것이다
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.7.0+cu128 True   <- 이렇게 나와야 한다
```

`/usr/bin/python` 이 나오거나 `ENV=` 가 비어 있으면 설치가 안 된 것이다.
`bash /opt/isaaclab-steps/diag.sh` 로 어디까지 됐는지 본다.

> **`(isaac_lab)` 이 안 붙는데 에러도 안 나는 경우.** 거의 항상 "환경은 잡혀 있고
> 표시만 없는" 상태다. 컨테이너 entrypoint 가 **비대화형** 쉘에서 `isaac-env.sh` 를
> 먼저 source 하는데, 거기서 `conda activate` 는 성공하지만 PS1 이 없어 표시가 안
> 붙는다. 이어지는 대화형 bash 의 `.bashrc` 가 다시 source 해도
> `CONDA_DEFAULT_ENV` 가 이미 맞아서 activate 를 건너뛴다. 그래서 조용히 아무 일도
> 안 일어난다. `which python` 이 conda 경로를 가리키면 **그냥 진행하면 된다.**
>
> 스크립트는 고쳐 뒀다(대화형인데 PS1 에 표시가 없으면 다시 activate 한다). 다만
> `/opt/isaaclab-scripts/` 는 **이미지에 구워진 사본**이라 이미지를 다시 빌드해야
> 반영된다. 빌드 없이 쓰려면 레포 쪽 사본을 source 하면 된다:
> ```bash
> source /workspace/rospider/docker/isaaclab/scripts/isaac-env.sh
> ```

> **`python` 과 `isaaclab -p` 는 같다.** `isaaclab` 은 `/opt/IsaacLab/isaaclab.sh` 의
> alias 고, `-p` 는 `${CONDA_PREFIX}/bin/python "$@"` 를 실행할 뿐이다. torch 를
> 재설치하는 `ensure_cuda_torch()` 는 `-i`(설치) 경로에서만 돈다. `(isaac_lab)` 이
> 활성화돼 있으면 **그냥 `python` 을 쓰면 된다.**
>
> **`isaac-shell` 안에서 ROS 를 source 하지 말 것.** python 3.10 심볼이 섞여
> Kit 기동이 깨진다. 이 과제는 ROS 를 한 줄도 쓰지 않는다.

### 5.3 준비물 설치 (최초 1회)

`/opt/conda` 가 named volume 이라 **한 번만 깔면 컨테이너를 지워도 남는다.**

```bash
pip install xacro matplotlib
```

| 패키지 | 왜 | 없으면 |
|---|---|---|
| `xacro` | xacro → URDF 전개 | 1단계에서 바로 멈춘다 |
| `matplotlib` | 4패널 결과 PNG | PNG 를 건너뛴다(`.npz` 와 CSV 는 그대로 나온다) |

### 5.4 Kit 없이 되는 단계 — 여기부터 순서대로

Kit(Isaac Sim) 을 띄우지 않는 단계를 앞에 몰아놨다. **문제가 생겼을 때 시뮬레이터
탓인지 로직 탓인지 바로 갈린다.** 각 단계의 "성공 표시" 를 보고 다음으로 간다.

#### 0단계 — 로봇 에셋 받기 (약 1분, 20 MB)

```bash
bash tasks/sensor_fusion/00_fetch_description.sh
```

성공 표시:

```
[3/3] 확인
  xacro  : 9 개
  메시   : 36 개
```

받아온 것은 `assets/rospider_description/` 에 들어간다. `.gitignore` 에 있으므로
커밋되지 않는다(Hiwonder 저작물이다).

#### 1단계 — xacro → URDF (수초)

```bash
python tasks/sensor_fusion/01_xacro_to_urdf.py
```

성공 표시:

```
[완료] assets/rospider_description/rospider.urdf (45378 bytes)
[확인] link 51개, joint 50개 (revolute 29개)
[확인] revolute 조인트: coxa_LF_joint, coxa_LM_joint, ... tibla_RR_joint
[확인] 참조된 메시 파일 전부 존재
```

revolute 29개 = 다리 18 + 팔 5 + 그리퍼 6. 이 숫자가 다르면 xacro 가 바뀐 것이다.
`[경고] 참조는 있는데 파일이 없는 메시` 가 뜨면 0단계를 다시 돌린다.

#### 2단계 — 센서 외부 파라미터 확인 (즉시)

```bash
python tasks/sensor_fusion/urdf_fk.py --joints joint2=0.85,joint3=-1.60,joint4=-1.26
```

성공 표시 — 아래 행렬이 `fusion.py` 의 `FusionParams.r_cam_imu` 기본값과 **같아야 한다**:

```
depth_cam_frame   [0.13125 0.0013  0.25086]      [-0.38422  0.593616 -0.593607  0.384222]

depth_cam_frame 축이 base_link 에서 가리키는 방향:
  Z(view)  -> [ 0.9123  0.     -0.4095]        <- 전방 24.17도 하향

R_cam<-imu (융합 코드에 넣을 상수):
[[ 1.        0.        0.      ]
 [ 0.       -0.409498 -0.912311]
 [ 0.        0.912311 -0.409498]]
```

팔 자세를 바꿀 거면 `--joints` 값을 바꿔 돌리고, 나온 값을 `fusion.py` 와
`rospider_cfg.py` 의 `CAM_*` 상수에 **같이** 반영한다. 한쪽만 바꾸면 조용히 틀린다.

#### 3단계 — 융합 로직만 검증 (수초)

```bash
python tasks/sensor_fusion/test_fusion.py
```

성공 표시 — 마지막 줄이 `전부 통과` 다. 7절의 표가 이 출력이다.
**여기서 실패하면 Isaac 으로 넘어가지 말자.** 융합 수식이나 상수가 틀린 것이다.

### 5.4.1 어떤 스크립트가 창(GUI)을 띄우나

**대부분은 창이 안 뜨는 게 정상이다.** 창이 안 뜬다고 실패한 게 아니다.

| 스크립트 | Isaac Sim | 창이 뜨나 |
|---|---|---|
| `00_fetch_description.sh` | 안 씀 | ✗ (git clone 일 뿐) |
| `01_xacro_to_urdf.py` | 안 씀 | ✗ (XML 처리일 뿐) |
| `urdf_fk.py` | 안 씀 | ✗ (numpy 계산일 뿐) |
| `test_fusion.py` | **안 씀** | ✗ — Kit 을 아예 import 하지 않는다 |
| `02_urdf_to_usd.py` | 띄움 | ✗ 기본은 헤드리스 |
| `02_urdf_to_usd.py --view` | 띄움 | **✓** |
| `run_fusion_demo.py` | 띄움 | ✗ 기본은 헤드리스 |
| `run_fusion_demo.py --gui` | 띄움 | **✓** |
| `run_all_conditions.sh` | 띄움 | ✗ 전부 헤드리스 |

**창을 보고 싶으면 `--view` 또는 `--gui` 를 붙여야 한다.** 기본을 헤드리스로 둔 이유는
GUI 렌더러가 VRAM 을 3~4 GB 먼저 먹기 때문이다. 8 GB 에서는 이게 크다.

### 5.4.2 `--view` / `--gui` 를 붙였는데도 창이 안 뜨면

순서대로 확인한다.

```bash
# ① 호스트 터미널에서
echo $DISPLAY                 # 보통 :0 또는 :1. 비어 있으면 X 세션이 아니다
xhost +local:root             # 매 로그인마다 1회 필요

# ② 컨테이너 안에서
echo $DISPLAY                 # 호스트와 같은 값이어야 한다
ls /tmp/.X11-unix/            # X0 같은 소켓이 보여야 한다
```

| 증상 | 원인과 해법 |
|---|---|
| 컨테이너의 `$DISPLAY` 가 비어 있음 | `docker compose run` 을 **`$DISPLAY` 가 설정된 터미널에서** 실행해야 한다. compose 가 `DISPLAY=${DISPLAY:-:0}` 로 넘긴다. ssh 로 붙었다면 X 포워딩이 없어서 그렇다 |
| `/tmp/.X11-unix/` 가 비어 있음 | 호스트에 X 서버가 없다. 순수 Wayland 세션이면 Xwayland 가 떠 있는지 확인 |
| `Authorization required, but no authorization protocol specified` | 호스트에서 `xhost +local:root` 를 안 했다. **컨테이너를 띄우기 전에** 해야 한다 |
| 창은 뜨는데 까맣게만 나옴 | 셰이더 컴파일 중이다. 첫 실행은 5~10분 기다린다 |
| `Failed to create a vulkan device` / GPU 관련 | 호스트에서 `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi` 가 되는지 먼저 확인 |

창 없이도 과제는 **전부 완성된다.** 결과 PNG 4패널(`outputs/sensor_fusion/*.png`)이 센서
출력과 판정을 다 담고 있어서, 노션 노트의 "실험 결과" 는 그걸로 채울 수 있다. 창은
"장면 스크린샷" 한 장과 다리 자세 눈대중에만 쓴다.

### 5.5 Kit 을 띄우는 단계

여기서부터 Isaac Sim 이 뜬다. **첫 실행은 셰이더 컴파일로 5~10분 멈춘 듯 보인다.**
정상이다. 두 번째부터는 캐시(`isaac-cache-ov` 볼륨)가 있어 빠르다.

#### 4단계 — URDF → USD 변환 (첫 실행 5~10분, 이후 1~2분)

```bash
python tasks/sensor_fusion/02_urdf_to_usd.py
```

성공 표시:

```
[완료] USD: /workspace/rospider/assets/usd/rospider.usd

[리짓바디 24개] base_link, coxa_LF, coxa_LM, ..., link1, link2, link3, link4, link5
[움직이는 조인트 29개] coxa_LF_joint, ..., joint1, ..., tibla_RR_joint
```

**이 출력의 바디 이름 목록을 꼭 보자.** `base_link` 가 있으면 `rospider_cfg.py` 의
센서 prim 경로(`{ENV_REGEX_NS}/Robot/base_link/depth_cam`) 가 맞다. 없고
`base_footprint` 같은 다른 이름이면 그 이름으로 바꿔야 한다.

눈으로 확인 (GUI):

```bash
python tasks/sensor_fusion/02_urdf_to_usd.py --view
```

창이 떠서 로봇이 보여야 한다. **여기서 다리 자세를 잡는다** — 매뉴얼 0절에 적었듯
다리 각도는 미검증이라, 메시가 바닥을 뚫거나 다리가 이상하게 꺾이면
`rospider_cfg.py` 의 `joint_pos` 에서 `coxa_/femur_/tibla_` 값을 조금씩 바꿔가며
다시 띄운다. base 가 고정이라 넘어지지는 않는다.

#### 5단계 — 융합 시뮬레이션

조건 하나씩 돌린다. 각 실행이 Kit 기동 포함 1~3분이다.

```bash
# 조건 1: 수평
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0 --tag level

# 조건 2: 20도 숙임 (장애물·임계값 동일)
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 20 --tag pitch20

# 조건 3: 넘어갈 수 있는 2 cm 단차
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0 \
       --obstacle_height 0.02 --tag lowstep

# 조건 4(선택): 바닥 추정을 끄고 설치 높이 상수를 믿게 해 본다 — 의도된 실패
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 20 \
       --floor_mode assumed_height --tag pitch20_assumed
```

> `--enable_cameras` 는 `run_fusion_demo.py` 가 코드에서 강제로 켜므로 안 쳐도 돌아간다.
> 그래도 적어 두는 게 좋다. **직접 새 스크립트를 쓸 때 이걸 빼먹으면** 뎁스가 빈
> 텐서로 와서 한참 헤맨다.

성공 표시 — 조건마다 한 줄씩 찍힌다:

```
[설정] pitch=20.0deg, 장애물 높이=0.15 m, floor_mode=estimated, 카메라 높이=0.318 m (수평일 때 0.367 m)
[임계값] DANGER<0.3 m, WARN<0.6 m, 기울기>25.0deg, 충격>6.0 m/s^2

장애물 0.90 m -> tilt= 20.0deg acc= 0.03 d_fused= 0.901 ... | fused=CLEAR  naive=DANGER  roi=WARN
장애물 0.70 m -> ...
  그림 저장: outputs/sensor_fusion/pitch20_d0.70.png
...
[완료] 로그 outputs/sensor_fusion/pitch20_log.csv (50 행)
```

세 열(`fused` / `naive` / `roi`) 이 **엇갈리는 줄**이 과제에서 보여줄 장면이다.

#### 5.5.1 계단 씬과 실시간 패널 (시연용)

`--scene` 으로 무엇을 놓을지 고른다.

| 값 | 무엇 | 쓰임 |
|---|---|---|
| `box` (기본) | 빨간 상자 하나. 실행 중에 옮길 수 있다 | 거리 스윕 실험 |
| `stairs` | 계단 4단 (5/10/15/20 cm) | **시연.** 높이맵이 띠로 갈라져 융합 과정이 보인다 |
| `mixed` | 넘어갈 수 있는 2 cm 단차 + 계단 | 시연. "멈출 것 / 넘어갈 것" 구분까지 보여준다 |

`--gui` 로 띄우면 창 오른쪽에 **실시간 패널**이 붙는다. `isaaclab.ui.widgets` 의
`ImagePlot` / `LinePlot` 으로 만든 것이라 Isaac Sim 창 안에서 바로 보인다.

```
┌──────────────────────────────┬─────────────────────────────┐
│                              │ DECISION: WARN              │
│                              │ ── DEPTH CAMERA (link4) ──  │
│  3D 뷰포트                    │   d_fused   0.550 m         │
│  로봇 + 계단 + 바닥            │   d_naive   0.415 m         │
│                              │   d_roi     0.740 m  ← 놓침  │
│  (IMU 가속도 화살표 표시)       │   obstacle pixels  2042     │
│                              │ ── IMU (base_link) ──       │
│                              │   body tilt      15.0 deg   │
│                              │   gravity in IMU (0,-0.26,  │
│                              │                    -0.97)   │
│                              │   camera down-tilt 24.2 deg │
│                              │ [Depth 이미지]               │
│                              │ [Height above ground] ← 핵심 │
│                              │ [Obstacle mask]             │
│                              │ [d_fused / d_roi 그래프]     │
└──────────────────────────────┴─────────────────────────────┘
```

가운데 **Height above ground** 패널이 "IMU 가 뎁스에 무슨 일을 하는가" 를 그대로
보여준다. 바닥은 0 근처(파랑), 계단은 단마다 다른 색 띠로 갈라진다.

시연 명령:

```bash
# 계단을 놓고 창을 띄운다. 패널 숫자가 실시간으로 갱신된다.
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --scene stairs

# 팔을 흔들어 카메라가 위아래를 훑게 한다 -> R_cam<-imu 가 매 스텝 바뀌는데도
# 융합이 계단 높이를 똑같이 잡아내는 걸 볼 수 있다
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui \
       --scene stairs --motion arm

# 넘어갈 수 있는 단차까지 같이
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --scene mixed
```

패널 생성에 실패해도 시뮬레이션은 계속 돈다(콘솔에 이유가 찍힌다).

#### 5.5.2 "Isaac Sim is not responding" 창이 뜰 때

**그냥 냅두면 된다. 계속 돌아간다.**

그 창은 Isaac Sim 이 띄운 게 아니라 **우분투 창 관리자(GNOME)** 가 띄운 것이다. 앱이
몇 초간 화면 갱신 요청에 응답하지 않으면 자동으로 뜨는데, Kit 은 셰이더 컴파일과 USD
로딩 중에 UI 스레드를 붙잡기 때문에 **정상 동작 중에도 뜬다.**

| 버튼 | 결과 |
|---|---|
| 아무것도 안 누름 | 그대로 계속 진행된다 ← 이게 정답 |
| **Wait** | 창만 닫힌다. 또 뜰 수 있다. 눌러도 되고 안 눌러도 된다 |
| **Force Quit** | 프로세스가 죽는다. **누르지 말 것** |

**살아 있는지는 터미널에서 판단한다.** 창이 멈춰 보여도 터미널에는 로그가 계속 찍힌다.
Stage 패널이 비어 있으면 아직 씬을 만드는 중이다.

첫 실행을 줄이는 방법:

```bash
# ① 같은 씬을 헤드리스로 한 번 돌려 셰이더 캐시(isaac-cache-ov 볼륨)를 채운다
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --scene stairs --tag warmup

# ② 그 다음 GUI. 두 번째부터는 훨씬 빠르다
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --scene stairs \
       --rendering_mode performance
```

`--rendering_mode` 는 AppLauncher 가 제공하는 인자로 `performance` / `balanced`(기본) /
`quality` 중 고른다. **VRAM 8 GB 에서는 `performance` 를 쓰자.**

콘솔에 `FabricManager::initializePointInstancer mismatched prototypes on point instancer:
/Visuals/Command/velocity_goal` 가 보이면 IMU 디버그 화살표 마커 때문이다. 무해하지만
거슬리면 `--no_imu_arrow` 로 끈다.

#### 6단계 — 스크린샷용 GUI 실행

노션 노트에 넣을 "장면 스크린샷" 은 여기서 찍는다.

```bash
# 호스트에서 한 번 (매 로그인마다)
xhost +local:root

# 컨테이너에서
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --pitch_deg 20
```

측정과 로그가 끝나면 창이 계속 떠 있다. 로봇·장애물·바닥이 보이는 각도로 돌려
스크린샷을 찍고, 창을 닫으면 종료된다. 창이 안 뜨고
`Authorization required, but no authorization protocol specified` 가 보이면
호스트에서 `xhost +local:root` 를 안 한 것이다.

### 5.6 한 번에 다 돌리기

조건 네 개를 순서대로 돌리는 스크립트를 넣어 뒀다. Kit 기동이 조건마다 한 번씩
들어가 10분 내외 걸린다.

```bash
bash tasks/sensor_fusion/run_all_conditions.sh
```

### 5.7 결과는 어디에

```
outputs/sensor_fusion/
├── level_log.csv            조건별 전체 로그 (노션 표의 원본)
├── level_d0.90.png          RGB / Depth / IMU 보정 높이 / 융합 마스크 4패널
├── level_d0.90.npz          원본 텐서 (depth, height, mask, rgb, intrinsics, IMU)
├── pitch20_log.csv
├── pitch20_d0.90.png
└── ...
```

레포가 bind mount 라서 **호스트 레포의 같은 경로에 그대로 있다.** 노션에는 PNG 를
올리고, CSV 는 표로 옮기거나 파일째 첨부한다.

`.npz` 를 다시 그리거나 임계값을 바꿔 재계산하려면 (Kit 불필요):

```python
import numpy as np
d = np.load("outputs/sensor_fusion/pitch20_d0.70.npz")
print(d["depth"].shape, d["projected_gravity_b"], d["lin_acc_b"])
```

### 5.8 다른 터미널에서 VRAM 보기

8 GB 에서 돌리는 중이니 한 번쯤 봐 두면 좋다. **호스트에서** 새 터미널을 열고:

```bash
watch -n 2 nvidia-smi
```

여유가 없으면 이 순서로 줄인다:

```bash
--no_rgb                                  # RGB 끄기 (제일 효과 큼)
--cam_width 96 --cam_height 72            # 해상도 줄이기
# --gui 를 빼고 헤드리스로 (GUI 렌더러가 3~4 GB 를 먼저 먹는다)
```

### 5.9 컨테이너를 나갔다 다시 들어올 때

```bash
exit                                                                    # 컨테이너 밖으로

# 다시 들어가기 (호스트)
docker compose -f docker/isaaclab/docker-compose.yml run --rm base

# 컨테이너 안 — 이 두 줄만 다시
source /opt/isaaclab-scripts/isaac-env.sh
cd /workspace/rospider
```

`pip install` 한 것, 받아온 에셋, 변환한 USD, 결과물은 **전부 남아 있다**
(앞의 둘은 도커 볼륨, 뒤의 둘은 레포 bind mount).

> **코드를 갱신할 때는 호스트 터미널에서 `git pull` 하자.** 컨테이너는 root 로 돌아서
> `detected dubious ownership` 로 막히고, 우회해서 pull 하면 새 파일이 root 소유가 된다.
> bind mount 라 호스트에서 받으면 컨테이너에 즉시 보인다.

### 5.10 체크포인트 — 여기까지 됐으면 다음으로

| # | 명령 | 이게 보이면 통과 |
|---|---|---|
| 0 | `00_fetch_description.sh` | `메시 : 36 개` |
| 1 | `01_xacro_to_urdf.py` | `revolute 29개` + `메시 파일 전부 존재` |
| 2 | `urdf_fk.py` | `R_cam<-imu` 가 `fusion.py` 기본값과 일치 |
| 3 | `test_fusion.py` | `전부 통과` |
| 4 | `02_urdf_to_usd.py` | `[리짓바디 ...]` 에 `base_link` 가 있음 |
| 4' | `02_urdf_to_usd.py --view` | 로봇이 바닥 위에 제대로 서 있음 |
| 5 | `run_fusion_demo.py` | `[완료] 로그 ...csv` + PNG 생성 |
| 6 | `--gui` 실행 | 스크린샷 확보 |

0~3 은 Kit 이 없어도 되니, 설치가 아직 안 끝났어도 **지금 바로 돌려볼 수 있다.**

## 6. 파일별로, 왜 그 코드인가

### `01_xacro_to_urdf.py` — xacro 전개

xacro 파일의 `$(find rospider_description)` 는 ament 패키지 조회다. 우리 쪽에는
ament 가 없으므로 **전개 전에 경로를 직접 치환**한다. 주의점 하나:

> `$(find pkg)/urdf/...` (include 경로) 는 **치환본이 있는 임시 디렉터리로**,
> 그 외(메시 경로) 는 원본 패키지 절대경로로 보내야 한다. include 를 원본으로
> 보내면 치환 안 된 파일이 다시 로드돼 `No module named 'ament_index_python'` 로 죽는다.

메시는 URDF 파일 기준 **상대경로**로 쓴다(`--mesh-mode relative`). 컨테이너 경로가
바뀌어도 안 깨진다.

### `urdf_fk.py` — 외부 파라미터 뽑기

numpy 만 쓴다. URDF 조인트 체인을 거슬러 올라가 `base_link` 기준 포즈를 곱한다.
쓰는 이유는 두 가지다.

1. **카메라가 base 가 아니라 팔 끝(link4) 에 달려 있다.** 그 체인을 손으로 계산하면
   틀린다. 조인트 6개(joint1~4 + camera_connect + depth_cam) 를 타고 가야 한다.
2. 팔 조인트를 0 으로 두면 **카메라가 천장을 본다.** 전방을 보는 조합을 찾아야 한다.
   이 스크립트로 찾은 값이 `joint2=0.85, joint3=-1.60, joint4=-1.26` 이고, 그때
   시선이 24.17도 하향이다. 세 조인트 합이 `-(π/2 + 하향각)` 이면 된다.

출력 중 가장 중요한 줄:

```
R_cam<-imu (융합 코드에 넣을 상수):
[[ 1.        0.        0.      ]
 [ 0.       -0.409498 -0.912311]
 [ 0.        0.912311 -0.409498]]
```

이게 **센서 캘리브레이션**이다. 실기라면 치구로 재야 하는 값이고, 시뮬에서는 URDF 가
정답을 알고 있다. **팔 자세를 바꾸면 반드시 다시 뽑아야 한다.**

### `fusion.py` — 융합 (Kit 의존 없음)

`omni` 를 import 하지 않는 것이 의도다. 덕분에 `AppLauncher` 없이 테스트할 수 있고,
**융합 버그와 시뮬레이터 문제를 분리**할 수 있다. 수식은 이렇다.

1. **역투영**: `z = depth[v,u]`, `x = (u−cx)/fx·z`, `y = (v−cy)/fy·z`
2. **중력 옮기기**: `g_cam = R_cam←imu · g_imu`, `up = −g_cam`
3. **축 만들기**: `forward = normalize(e_z − (e_z·up)·up)`, `right = normalize(forward × up)`
   (수평일 때 `forward=(0,0,1)`, `up=(0,−1,0)` → `right=(1,0,0)` 으로 맞는지 확인했다)
4. **높이**: `h = p·up − ground`.
   `ground` 를 구하는 방식이 두 가지고, 이게 분석거리가 된다.
   - `floor_mode="estimated"` (기본): 보이는 점의 `p·up` 하위 5 백분위수를 바닥으로.
     설치 높이를 몰라도 되고, 숙여서 카메라가 낮아져도 따라간다.
   - `floor_mode="assumed_height"`: 설치 높이 상수를 믿는다. 단순하지만 기울면 밀린다.
5. **마스크**: `0.03 < h < 0.40`, `|p·right| < 0.15`, `0.05 < p·forward < 2.0`, `isfinite`
6. **거리**: `d_fused = min(p·forward | 마스크)`, 마스크 픽셀이 25개 미만이면 `inf`
7. **판정**: 거리 등급 위에 IMU 상위 조건(기울기/충격) 을 덮어쓴다

비교군을 **두 개** 같이 낸다. 뎁스 단독(`d_naive`, 중앙 밴드 최소 뎁스)은 사실상
허수아비라서, **고정 ROI**(`d_roi`, 화면 위쪽 절반만 봄) 도 넣었다. 수평에서는
고정 ROI 가 제대로 동작하기 때문에, "그래도 IMU 가 필요한가?" 라는 질문에
정면으로 답할 수 있다.

### `rospider_cfg.py` — 로봇·센서·씬

판단이 들어간 부분만 짚는다.

**`fix_base=True` 로 변환한다.** 이건 보행 과제가 아니라 인식 과제다. base 를 고정하면
(a) 다리 게인을 못 잡아 로봇이 주저앉는 사고가 사라지고, (b) 몸체 기울기를 **정확히
원하는 값**으로 줄 수 있어 조건 비교가 깔끔해진다. 보행까지 가려면 `False` 로 바꾸고
다리 PD 부터 다시 잡아야 한다.

**센서는 둘 다 `base_link` 에 붙인다.** `merge_fixed_joints=True`(기본) 로 변환하면
`imu_link`·`depth_cam_link` 가 부모에 흡수돼 **prim 이 사라진다.** 그 경로로 센서를
걸면 이렇게 죽는다:

```
RuntimeError: Failed to find a prim at path expression: .../imu_link
```

그래서 `base_link` 에 걸고 URDF 오프셋을 `OffsetCfg` 로 직접 준다. `Imu` 는 대상
prim 이 **RigidBody 여야** 하므로(`UsdPhysics.RigidBodyAPI` 를 확인한다) 더미 링크에
걸 수도 없다. `--no-merge-fixed-joints` 로 변환하면 `imu_link` 가 살지만, 질량 없는
더미 링크들 때문에 PhysX 가 투덜댄다.

**팔을 단단히 붙잡는다** (`stiffness=200, damping=20`). 팔이 흔들리면 `R_cam←imu` 가
틀려지고, 그 오차는 바닥 제거 전체를 망친다.

**URDF 의 `effort=1 N·m`, `velocity=1 rad/s` 는 과소값이다.** base.urdf.xacro 의
`default_joint_effort`/`default_joint_velocity` 가 둘 다 1 로 박혀 있다. 실기 서보는
35 kg·cm ≈ 3.4 N·m 다. `ImplicitActuatorCfg.effort_limit_sim` 으로 덮었다.

**장애물은 `RigidObjectCfg` + `kinematic_enabled=True`.** 중력에 떨어지지 않으면서
`write_root_pose_to_sim()` 으로 **실행 중에 옮길 수 있다.** 거리 조건마다 Kit 을
다시 띄우지 않아도 되므로 실험이 몇 분에서 몇 초로 줄어든다.

### `run_fusion_demo.py` — 본체

- `args_cli.enable_cameras = True` 를 **코드에서 강제**한다. 카메라 센서가 있는 씬은
  헤드리스에서도 이 플래그가 필요한데, 빼먹으면 뎁스가 빈 텐서로 와서 한참 헤맨다.
- import 순서를 지킨다. `AppLauncher` 로 앱을 띄운 **뒤에야** `isaaclab.sim`,
  `isaaclab.scene`, `isaaclab.sensors` 를 import 할 수 있다.
- `--dump_asset_info` 로 `robot.body_names` / `robot.joint_names` 만 찍고 끝낼 수 있다.
  센서 prim 경로가 안 맞을 때 여기부터 본다.
- 조건마다 CSV 한 줄씩, 그리고 **RGB / Depth / IMU 보정 높이 / 융합 마스크** 4패널
  PNG 를 남긴다. 노션 노트의 "센서 출력과 최종 판단을 보여주는 이미지" 가 이거다.
- 원본은 `.npz` 로도 남긴다. 그림을 다시 그리거나 임계값을 바꿔 재계산할 때 쓴다.

## 7. 실험 조건과 결과

`test_fusion.py` 를 합성 뎁스로 돌린 **실제 출력**이다. Isaac 안에서 돌리면 메시
형상과 렌더 노이즈 때문에 소수점은 달라지지만, 판정 패턴은 같아야 한다.

| 조건 | 기울기 | 장애물 | d_fused | d_naive | d_roi | **융합** | 뎁스단독 | 고정ROI |
|---|---|---|---|---|---|---|---|---|
| A 수평 | 0.0° | 0.80 m, h=0.15 | **0.800** | 0.415 | 0.820 | CLEAR | WARN ✗ | CLEAR |
| B 20도 숙임 | 20.0° | 0.80 m, h=0.15 | **0.800** | 0.287 | 0.442 | CLEAR | DANGER ✗ | WARN ✗ |
| C1 접근 | 0.0° | 0.55 m, h=0.15 | **0.550** | 0.415 | 0.592 | WARN | WARN | WARN |
| C2 접근 | 0.0° | 0.25 m, h=0.15 | **0.250** | 0.317 | 0.905 | DANGER | WARN ✗ | CLEAR ✗ |
| D 낮은 단차 | 0.0° | 0.55 m, **h=0.02** | **inf** | 0.415 | 0.905 | CLEAR | WARN ✗ | CLEAR |
| E 넘어짐 | 30.0° | 0.80 m | inf | 0.240 | 0.331 | **STOP_TILT** | DANGER | WARN |
| F 충격 | 0.0° | 0.80 m, a=8 m/s² | 0.800 | 0.415 | 0.820 | **STOP_IMPACT** | WARN | CLEAR |
| G B를 상수높이로 | 20.0° | 0.80 m, h=0.15 | 0.102 ✗ | 0.287 | 0.442 | DANGER ✗ | DANGER | WARN |
| H 계단 4단 | 0.0° | 5/10/15/20 cm | **0.550** | 0.415 | 0.740 | WARN | WARN | **CLEAR ✗** |
| I 계단 + 15° 숙임 | 15.0° | 같음 | **0.550** | 0.313 | 0.516 | WARN | WARN | WARN |

읽는 법:

- **A/B 쌍이 이 과제의 핵심이다.** 장애물은 그대로인데 몸만 20도 숙였다. 융합은
  `0.800 m` 를 그대로 유지했고(오차 0), 고정 ROI 는 `0.820 → 0.442` 로 무너져
  CLEAR 가 WARN 이 됐다. **잘라내기로는 못 막고, 중력을 알아야 막힌다.**
- **C2 는 뎁스 단독이 왜 위험한지 보여준다.** 장애물이 0.25 m 까지 왔는데 뎁스 단독은
  `0.317 m` → WARN 에 머문다. 최소 뎁스가 **사선 거리**라서 수평거리보다 크게 읽히기
  때문이다. 융합은 중력 정렬 전방거리를 쓰므로 `0.250` 으로 정확하다.
- **D 는 '지나갈 수 있는 것' 을 가려낸다.** 2 cm 단차는 h=0.02 < 0.03 이라 마스크에서
  빠지고 `inf` 가 된다. 로봇이 멈출 이유가 없다. 거리만 보는 쪽은 구분을 못 한다.
- **E/F 는 IMU 단독 기여다.** 거리와 무관하게 자세·충격이 상위 정지를 건다.
- **H 는 고정 ROI 가 반대로 틀리는 경우다.** 지금까지는 고정 ROI 가 바닥을 장애물로
  **오인(false positive)** 했는데, 계단에서는 화면 위쪽만 보느라 **계단을 통째로 놓친다
  (false negative)**. CLEAR 를 내는 쪽이 오경보보다 위험하다. 융합은 0.550 m 로 첫 단을
  정확히 잡는다.
- **H 의 높이맵이 이 과제의 그림이다.** 장애물로 분류된 점들의 높이가
  0.031 ~ 0.200 m 범위에 **띠 네 개**로 갈라진다(1단 758 px / 2단 614 / 3단 424 / 4단 246).
  "IMU 로 중력 정렬을 했더니 계단이 높이별로 분리됐다" 가 눈에 보인다.
- **I 는 기울여도 높이 추정이 안 흔들린다는 증거다.** 15° 숙여도 거리는 0.550 m 그대로고,
  1~3단이 같은 높이 구간에 그대로 잡힌다. 다만 **4단은 246 px → 0 px 로 사라지는데,
  이건 융합 오류가 아니라 시야(FOV) 한계다** — 하향 24° 장착에 몸이 15° 더 숙으면
  먼 곳과 높은 곳이 화면 위로 잘려 나간다. 노트의 '한계' 항목에 쓸 재료다.
- **G 는 의도된 실패다.** 설치 높이를 상수로 믿으면, 20도 숙일 때 카메라가 실제로
  6 cm 낮아진 걸 모른 채 모든 점의 높이를 그만큼 높게 본다. 바닥이 장애물로 올라와
  `0.102 m` DANGER 오경보가 난다. **바닥을 추정하는 쪽(기본값) 이 왜 필요한지** 가
  이 한 줄로 설명된다. 노트의 '개선 방향' 에 그대로 쓸 수 있다.

조건을 더 만들려면: `--dists` 로 거리 스윕(한 실행에 여러 지점), `--pitch_deg` 로 기울기,
`--obstacle_height` 로 높이, `--floor_mode` 로 바닥 추정 방식.

## 8. 결과 분석 — 노트에 쓸 뼈대

**단일 센서 대비 장점**

- 바닥을 기하학적으로 제거하므로, 카메라가 어디를 보든 임계값을 다시 안 잡아도 된다.
- 거리 기준이 "사선 거리" 가 아니라 "중력 정렬 수평거리" 라서 임계값이 물리적 의미를
  갖는다 (0.30 m = 로봇 앞 30 cm).
- 물체 **높이**를 알 수 있으므로 "멈출 것 / 넘어갈 것" 을 구분한다. 거리만으로는 못 한다.
- 자세·충격이라는, 카메라로는 못 보는 위험을 같은 판정 변수에 넣을 수 있다.

**한계 (실제로 있는 것만)**

1. **사각지대.** 장착 기하 때문에 카메라 앞 0.29 m 안쪽 바닥이 안 보인다. 그보다
   가까운 낮은 장애물은 원리적으로 못 본다.
2. **바닥 추정이 바닥을 봐야 된다.** `floor_mode="estimated"` 는 화면에 바닥이
   있다고 가정한다. 벽에 붙어 서면 하위 5 백분위수가 벽이 되어 전체 높이가 밀린다.
3. **단일 평면 가정.** 계단이나 경사 바닥에서는 "바닥 한 장" 가정이 깨진다.
4. **측정 시점.** 두 센서를 같은 스텝에 읽어 시점 차이를 0 으로 뒀지만, 실기에서는
   IMU 가 수백 Hz, 뎁스가 30 Hz 다. 몸이 빠르게 끄덕일 때 뎁스 한 장 안에서 자세가
   변해 롤링 왜곡이 생긴다. 시뮬에서 재현하려면 카메라 `update_period` 를 키워보면 된다.
5. **IMU 드리프트가 없다.** Isaac 의 `Imu` 는 바이어스·노이즈가 없는 이상적 센서다.
   실기 자세 추정은 자이로 적분 드리프트와 가속도계 노이즈를 상보 필터/EKF 로
   섞어야 한다. 시뮬 결과가 실기보다 좋게 나오는 가장 큰 이유다.

**개선 방향**

- 바닥을 백분위수 대신 **RANSAC 평면 적합**으로 추정 → 경사 바닥과 계단에 대응.
- IMU 에 인위적 노이즈·바이어스를 주입하고 상보 필터를 붙여 실기에 가깝게.
- 뎁스를 중력 정렬 좌표로 옮겼으면 **2.5D 높이맵**으로 누적 → 단일 프레임 노이즈에
  덜 흔들리고, 발 디딜 곳 판단까지 확장된다.
- LiDAR 를 더해 3센서로. RosPider 는 `lidar_link` 도 URDF 에 갖고 있다
  (base_link 기준 (0.10163, 0.00022, 0.03515) m). `RayCaster` 로 붙일 수 있다.
- 다수 환경 학습까지 가면 `Camera` 를 `TiledCamera` 로 교체 (VRAM 효율).

## 9. 노션 노트 작성 체크리스트

| 노션 항목 | 무엇으로 채우나 |
|---|---|
| 기본 정보 | 제목 `[센서융합 과제] 학번_이름_RosPider 뎁스+IMU 근접 판단`, 학번·이름 |
| 과제 목표 | 2절 그대로. "바닥을 장애물로 오인하는 문제" 를 앞에 세우면 이야기가 선다 |
| 실행 환경 | **5.0.2절의 사양 표를 그대로 옮긴다.** 컨테이너가 Ubuntu 22.04, 호스트가 24.04 인 이유도 한 줄 적으면 좋다 |
| 장면 스크린샷 | `--gui` 로 띄워 캡처 + `02_urdf_to_usd.py --view` 의 로봇 전체 모습 |
| 센서 위치·방향·설정 | 4절 표를 그대로. `urdf_fk.py` 출력을 캡처해 붙이면 근거까지 보인다 |
| 융합 흐름도 | 3절 다이어그램 |
| 핵심 코드 | `fusion.py` 의 `fuse()` 2~7단계. 역투영·중력 옮기기·바닥 제거 세 토막이면 충분하다 |
| 실험 결과 | 7절 표 + `outputs/sensor_fusion/*.png` 4패널 그림 (조건별로 최소 2장) |
| 결과 분석 | 8절. 특히 A/B 쌍과 조건 G |
| 발생한 오류와 해결 | 10절에서 실제로 밟은 것만 골라 쓴다 |
| 첨부·참고 | 이 레포 링크, `tasks/sensor_fusion/` 경로, 11절 출처 |

**제출 전 확인**: 노션 페이지 공유를 "링크를 아는 모든 사람 — 읽기 가능" 으로 바꿀 것.
이미지·코드 링크가 로그아웃 상태에서 열리는지 시크릿 창으로 확인할 것.

## 10. 터지는 지점과 대처

레포 루트 `CLAUDE.md` 의 함정 표를 먼저 보고, 이 과제에서 추가로 만날 것들은 이렇다.

| 증상 | 원인과 해법 |
|---|---|
| `No module named 'ament_index_python'` (xacro 전개 중) | include 가 치환 안 된 원본을 다시 읽었다. `01_xacro_to_urdf.py` 가 `$(find ...)/urdf/` 를 임시 디렉터리로 돌리는 부분을 건드리지 말 것 |
| `Failed to find a prim at path expression: .../imu_link` | `merge_fixed_joints=True` 로 흡수됐다. `base_link` 에 붙이고 `OffsetCfg` 를 주거나 `--no-merge-fixed-joints` 로 변환 |
| `Failed to find a RigidBodyAPI for the prim paths` | `Imu` 를 더미/비강체 prim 에 걸었다. 강체 prim 에 걸어야 한다 |
| 뎁스가 전부 `inf` | `--enable_cameras` 를 빼먹었거나, `clipping_range` 의 far 가 너무 짧거나, 카메라가 허공을 본다. `--gui` 로 시선을 확인 |
| `data_types` 에 뎁스가 없다 | `CameraCfg.data_types` 기본값은 `["rgb"]` 다. `"distance_to_image_plane"` 을 명시해야 한다 |
| 로봇이 바닥을 뚫거나 다리가 꺾인다 | 다리 조인트 목표가 0 이라 그렇다. GUI 로 보면서 coxa/femur/tibla 부호를 찾는다. `fix_base=True` 라 넘어지지는 않는다 |
| 카메라가 천장을 본다 | 팔 조인트가 0 이다. `ARM_OBSERVE_POSE` 를 적용했는지, 그리고 팔 게인이 자세를 붙잡는지 확인 |
| `fatal: detected dubious ownership in repository at '/workspace/rospider'` | 컨테이너는 root 로 도는데 파일은 호스트 사용자 소유다. **애초에 컨테이너 안에서 git 을 쓰지 말고 호스트 터미널에서 pull 하자**(bind mount 라 즉시 보인다). 컨테이너에서 pull 하면 새 파일이 root 소유가 돼 호스트에서 건드리기 번거로워진다. 꼭 컨테이너에서 써야 하면 `git config --global --add safe.directory /workspace/rospider` |
| `(isaac_lab)` 이 프롬프트에 안 붙고 에러도 없음 | 대개 환경은 잡혀 있고 표시만 없다. `which python` 이 `/opt/conda/envs/isaac_lab/bin/python` 이면 그냥 진행. 5.2절 참고 |
| `No module named 'omni'` | 고장 아니다. `AppLauncher` 뒤에서만 import 가능 |
| `No module named 'isaaclab'` | `isaaclab.sh -i` 가 실패를 종료코드에 안 싣는다. `bash /opt/isaaclab-steps/diag.sh` |
| VRAM OOM / 느림 | `--cam_width 96 --cam_height 72 --no_rgb`, `num_envs=1`, `--gui` 끄기 |
| `LLVM ERROR: out of memory` | VRAM 이 아니라 **시스템 RAM** 이다 |
| `"Isaac Sim 5.1.0" is not responding` 창 | **정상이다. 그냥 냅두면 계속 돌아간다.** GNOME 이 띄운 창이고, Kit 은 셰이더 컴파일 중 UI 스레드를 붙잡는다. Force Quit 만 누르지 말 것. 5.5.2절 |
| 첫 GUI 실행이 10분 넘게 걸린다 | 헤드리스로 먼저 한 번 돌려 셰이더 캐시를 채우고, `--rendering_mode performance` 를 쓴다. 5.5.2절 |
| `FabricManager::initializePointInstancer mismatched prototypes` | IMU 디버그 화살표 마커. 무해하다. `--no_imu_arrow` 로 끌 수 있다 |
| 창이 안 뜬다 | 대부분 정상이다. `--view` / `--gui` 를 붙여야 뜬다. 5.4.1절의 표를 보자 |
| `--view`/`--gui` 를 붙였는데도 안 뜬다 | 5.4.2절의 X11 점검 순서 |
| 창이 안 뜨고 `Authorization required...` | 호스트에서 `xhost +local:root`. **컨테이너를 띄우기 전에** 해야 한다 |
| `d_fused` 가 터무니없이 작다 | `R_cam<-imu` 가 팔 자세와 안 맞는다. `urdf_fk.py` 로 다시 뽑아 `FusionParams.r_cam_imu` 를 갱신 |

## 11. 참고 자료와 출처

- **RosPider 로봇 기술 파일**: `github.com/Hiwonder/ROSpider` →
  `src/simulations/rospider_description/` (xacro + STL). 이 과제의 모든 장착 수치가
  여기서 나왔다. 메시/xacro 는 Hiwonder 저작물이므로 레포에 커밋할 때 라이선스를 확인할 것.
- **Hiwonder ROSpider 공식 문서**: `docs.hiwonder.com/projects/ROSpider/en/jetson-orin-nano-version/`
  (하드웨어 구성, Jetson 이미지, ROS 2 Humble 기준)
- **제조사 배포 드라이브**: 튜토리얼 / 소프트웨어 / 시스템 이미지·소스 / 하드웨어 자료 4개 폴더.
  URDF 는 여기 없고 위 GitHub 에 있다.
- **Isaac Lab v2.3.1 소스** (API 근거):
  - `source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py`
  - `source/isaaclab/isaaclab/sensors/camera/camera_cfg.py`, `camera_data.py`
  - `source/isaaclab/isaaclab/sensors/imu/imu_cfg.py`, `imu.py`, `imu_data.py`
  - `scripts/tutorials/04_sensors/add_sensors_on_robot.py` (씬 구성 패턴의 원본)
- **RosPider 실기 ROS 배포판은 미확인.** 컨테이너는 Humble 이다. 보드에서
  `echo $ROS_DISTRO` 로 확인해야 DDS 상호운용을 말할 수 있다.
