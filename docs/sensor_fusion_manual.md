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

## 5. 실행 순서

```bash
# --- 호스트에서 (1회) ---
bash docker/isaaclab/host_setup.sh
xhost +local:root                      # GUI 쓸 거면 필수

docker compose -f docker/isaaclab/docker-compose.yml build base
docker compose -f docker/isaaclab/docker-compose.yml run --rm base

# --- 컨테이너 안 ---
bash /opt/isaaclab-steps/run_all.sh    # 이미 돼 있으면 건너뜀
bash /opt/isaaclab-steps/diag.sh       # 상태 점검
cd /workspace/rospider

# 0단계: 로봇 에셋 받기 (ROS 불필요)
bash tasks/sensor_fusion/00_fetch_description.sh

# 1단계: xacro -> URDF (ROS 불필요, Kit 불필요)
pip install xacro
python tasks/sensor_fusion/01_xacro_to_urdf.py
#   -> [확인] link 51개, joint 50개 (revolute 29개) / 메시 전부 존재

# 1.5단계: 센서 외부 파라미터 확인 (Kit 불필요)
python tasks/sensor_fusion/urdf_fk.py --joints joint2=0.85,joint3=-1.60,joint4=-1.26
#   -> 카메라/IMU 포즈와 R_cam<-imu 가 출력된다. fusion.py 기본값과 같아야 한다

# 1.9단계: 융합 로직만 먼저 검증 (Kit 불필요, 수초)
python tasks/sensor_fusion/test_fusion.py
#   -> "전부 통과" 가 나와야 다음으로 간다

# 2단계: URDF -> USD  (여기서부터 Kit)
python tasks/sensor_fusion/02_urdf_to_usd.py            # 헤드리스 변환
python tasks/sensor_fusion/02_urdf_to_usd.py --view     # 눈으로 확인

# 3단계: 융합 시뮬레이션. --enable_cameras 필수
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0  --tag level
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 20 --tag pitch20
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0 \
       --obstacle_height 0.02 --tag lowstep
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --pitch_deg 20   # 스크린샷용

# 결과: outputs/sensor_fusion/{tag}_log.csv, {tag}_d0.90.png, .npz ...
```

**ROS 는 한 줄도 안 쓴다.** `isaac-shell` 안에서 `/opt/ros/humble/setup.bash` 를 절대
source 하지 말 것 (3.10 심볼이 섞여 Kit 기동이 깨진다). RViz 로 보고 싶어지면
그때 `ros-shell` 을 따로 열고 표준 메시지로만 주고받자.

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
| 실행 환경 | Isaac Sim 5.1.0 / Isaac Lab 2.3.1 / Python 3.11 / PyTorch 2.7.0+cu128 / Ubuntu 22.04 컨테이너 / RTX 4060 Laptop 8 GB. `docker/isaaclab/` 참고 |
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
| `No module named 'omni'` | 고장 아니다. `AppLauncher` 뒤에서만 import 가능 |
| `No module named 'isaaclab'` | `isaaclab.sh -i` 가 실패를 종료코드에 안 싣는다. `bash /opt/isaaclab-steps/diag.sh` |
| VRAM OOM / 느림 | `--cam_width 96 --cam_height 72 --no_rgb`, `num_envs=1`, `--gui` 끄기 |
| `LLVM ERROR: out of memory` | VRAM 이 아니라 **시스템 RAM** 이다 |
| 창이 안 뜨고 `Authorization required...` | 호스트에서 `xhost +local:root` |
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
