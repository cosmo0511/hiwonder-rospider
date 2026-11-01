# 장애물 넘기 강화학습 매뉴얼

RosPider 가 실물 장애물 코스(과속방지턱, 구덩이, 굴러가는 파이프, 그레이팅 계단, 작업대)를
넘도록 Isaac Lab 2.3.1 + RSL-RL(PPO)로 학습하는 방법. 코드는 `tasks/obstacle_rl/` 에 있다.

> **아직 Isaac Lab 에서 한 번도 돌려 보지 않은 코드다** (2026-11-01 기준).
> 지형 기하만 numpy 검사(`test_course.py`)를 통과했다. 처음 돌리는 사람은 아래
> "처음 돌릴 때 확인할 것"부터 보고, 에러가 나면 메시지 전체를 공유해 주세요.

---

## 1. 준비 (최초 1회)

컨테이너 환경은 `docker/isaaclab/QUICKSTART.md` 를 따른다. 여기서는 그 뒤부터.

```bash
# 호스트에서 컨테이너 진입
docker compose -f docker/isaaclab/docker-compose.yml run --rm base
```

이하 전부 **컨테이너 안 `isaac-shell`**, 레포 루트(`/workspace/rospider`)에서 친다. ROS 는 source 하지 않는다.

```bash
# 1) 로봇 에셋 받기 (assets/ 는 git 에 없다)
bash tasks/sensor_fusion/00_fetch_description.sh

# 2) xacro -> URDF
pip install xacro
python tasks/sensor_fusion/01_xacro_to_urdf.py --pkg assets/rospider_description

# 3) URDF -> USD. 보행 학습은 몸이 움직여야 하므로 반드시 --no-fix-base
python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base
#    -> assets/usd/rospider_float.usd 가 생겨야 한다

# 4) RL 라이브러리 확인 (Isaac Lab 설치 때 같이 깔림)
python -c "import rsl_rl, isaaclab_rl; print('ok')"
#    안 되면: FORCE=1 ISAACLAB_RL_FRAMEWORK=rsl_rl bash /opt/isaaclab-steps/40_isaaclab.sh

# 5) 지형 검사 (Kit 안 띄움, 몇 초)
python tasks/obstacle_rl/test_course.py      # 마지막 줄 "전부 통과"
```

---

## 2. 학습

```bash
python tasks/obstacle_rl/train.py --headless --num_envs 64
```

| 옵션 | 뜻 |
|---|---|
| `--headless` | 창 없이. **학습은 반드시 헤드리스** (GUI 가 VRAM 3~4 GB 를 먼저 먹는다) |
| `--num_envs 64` | 동시에 학습하는 로봇 수. RTX 4060 8 GB 는 64 부터 시작해서 128, 256 으로 올려 보며 한계를 찾는다 |
| `--max_iterations 3000` | 학습 반복 수 (기본 3000) |
| `--height_curriculum` | 장애물을 낮은 높이(0.5 배)부터 실물 높이까지 키우며 학습. 기본은 실물 높이 고정 |
| `--resume <경로>/model_1000.pt` | 끊긴 학습 이어서 |
| `--seed 42` | 난수 시드 |

결과는 `logs/rsl_rl/rospider_obstacle/<날짜_시각>/` 에 쌓인다 (git 에 안 올라감).

- `model_<반복수>.pt` : 100 반복마다 저장되는 정책
- `params/env.yaml`, `params/agent.yaml` : 그 학습에 쓴 설정 전체

### 학습이 잘 되고 있는지 보기

다른 터미널(같은 컨테이너)에서:

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

| 볼 값 | 좋아지는 방향 |
|---|---|
| `Train/mean_reward` | 올라감 |
| `Train/mean_episode_length` | 올라감 (안 넘어지고 오래 버팀) |
| `Episode_Reward/forward_progress` | 올라감 (앞으로 나아감) |
| `Episode_Termination/base_contact` | 내려감 (몸통이 장애물에 덜 부딪힘) |
| `Curriculum/terrain_levels` | `--height_curriculum` 일 때만 의미 있음. 올라가면 더 높은 장애물로 진급 중 |

1000 반복이 지나도 `mean_reward` 가 제자리면 `--height_curriculum` 으로 다시 시작해 본다.

---

## 3. 평가 (실물 코스)

학습은 장애물을 섞은 칸에서 하고, 평가는 **실물과 같은 맵 1, 맵 2** 에서 한다.

```bash
# 창으로 보기 (맵 1, 2 각각 3번)
python tasks/obstacle_rl/play.py --checkpoint logs/rsl_rl/rospider_obstacle/<날짜>/model_3000.pt

# 숫자만 (맵마다 20번)
python tasks/obstacle_rl/play.py --checkpoint <...>.pt --headless --episodes 20

# 맵 하나만
python tasks/obstacle_rl/play.py --checkpoint <...>.pt --map 1
```

- 명령은 0.1 m/s 직진 고정, 마찰 / 질량 무작위화와 관측 잡음은 끈다.
- 작업대(높이 0.24 m)에 올라서면 성공. 결과 표: `outputs/obstacle_rl/play_<날짜>.csv`
  (`reached_platform`, `best_progress_m` = 출발점에서 가장 멀리 간 거리, `goal_m` = 작업대까지 거리)

---

## 4. 지형과 환경 한눈에

### 코스 (실물 치수, 단위 m)

| | 맵 1 | 맵 2 |
|---|---|---|
| 순서 | 구덩이 → 파이프 2개 → 그레이팅 계단 → 작업대 | 과속방지턱 → 파이프 2개 → 그레이팅 계단 → 작업대 |
| 출발선 ~ 작업대 끝 | 2.797 | 2.697 |

- 출발선 ~ 첫 장애물 0.60, 파이프 앞뒤 간격 0.15
- 과속방지턱 0.40 x 높이 0.08 (사다리꼴) / 구덩이 0.50 x 깊이 0.047 (둥근 단면)
- 파이프 Ø0.0486 x 1.0, 두 개, 간격 0.10, **굴러감** (질량 2.08 kg, 계산값)
- 계단 4단 (디딤 0.15, 한 단 0.06): 위 30 mm 는 그레이팅(구멍 12.7 x 70.8 mm, 위아래 관통)
- 작업대 0.60 x 높이 0.24, 끝에 ㄴ자 용접판 (가정 크기 0.20 x 0.60, 세운 판 0.15)

### 학습 맵

4.0 x 1.2 m 칸을 6행 x 14열로 깐다. 열마다 칸 종류가 정해져 있다.

| 칸 종류 | 비율 |
|---|---|
| 평지 | 8 % |
| 방지턱만 / 구덩이만 / 파이프만 | 각 14 % |
| 계단 + 작업대 | 18 % |
| 맵 1 / 맵 2 그대로 | 각 16 % |

로봇은 칸마다 출발선 뒤에서 출발해 +x 로 걷는다. 높이는 기본이 실물 고정(행이 달라도 같음).

### 정책이 받는 것 / 내는 것

| 구분 | 내용 |
|---|---|
| 관측 | 몸 선속도·각속도, 중력 방향(기울기), 속도 명령, 다리 18개 관절각·속도, 직전 행동, 몸 앞 0.8 x 0.6 m 높이 스캔 |
| 행동 | 다리 18개 목표 각도 (선 자세 + 0.25 x 출력, rad). 팔·그리퍼는 고정 |
| 명령 | 앞으로 0.04 ~ 0.12 m/s, 옆 ±0.02, 방향 유지 |
| 보상 | 속도 따라가기, 앞으로 나아간 만큼(+), 발을 0.3 s 이상 들기(+) / 위아래·기울기 흔들림, 토크, 행동 급변, 허벅지 충돌, 관절 한계(-) |
| 끝나는 조건 | 40 초, 몸통이 땅이나 장애물에 닿음, 1 rad 넘게 기울어짐 |
| 무작위화 | 마찰 0.5~1.0, 몸통에 0.8~1.8 kg 추가 (URDF 1.66 kg → 실물 3.2 kg), 출발 위치·방향, 약한 밀기, 파이프 위치 ±3 cm |

---

## 5. 바꾸고 싶을 때 어디를 고치나

| 바꿀 것 | 파일 / 위치 |
|---|---|
| 장애물 치수, 간격, 파이프 질량, 용접판 크기 | `course.py` 맨 위 상수 |
| 학습 맵에서 칸 종류 비율, 행·열 수 | `terrain_cfg.py` 의 `TRAIN_MIX`, `OBSTACLE_TERRAINS_CFG` |
| 보상 가중치, 명령 속도 범위, 무작위화 범위, 에피소드 길이 | `env_cfg.py` 의 `RoSpiderObstacleEnvCfg.__post_init__` |
| 평가 조건 (명령 속도 등) | `env_cfg.py` 의 `RoSpiderObstacleEnvCfg_PLAY` |
| 다리 게인(강성/감쇠), 최대 토크, 팔 자세 | `robot_cfg.py` |
| PPO (신경망 크기, 학습률, 반복 수) | `agents/rsl_rl_ppo_cfg.py` |

`course.py` 를 고쳤으면 `python tasks/obstacle_rl/test_course.py` 로 다시 검사한다
(설계 수치가 바뀌었으면 검사 기대값도 같이 고친다).

### 다리 게인이란

관절마다 "목표 각도로 가라"는 명령을 받으면 `힘 = 강성 x (목표 - 현재 각도) - 감쇠 x 관절 속도` 로 힘을 낸다.

- **강성(stiffness)**: 용수철 세기. 약하면 몸무게에 다리가 주저앉고, 세면 딱딱하게 튄다.
- **감쇠(damping)**: 쇼크업소버. 약하면 떨리고, 세면 굼뜨다.
- **최대 토크(effort_limit_sim)**: 모터 힘의 한계. 실물 서보 35 kg·cm = 3.4 N·m.

지금 값(강성 20 N·m/rad, 감쇠 1)은 `sensor_fusion` 과 같은 출발값이다. 다른 레포의 걷기 스크립트
(`rospider_wasd.py`, 실제로 걷는 걸 확인함)는 강성 약 57 N·m/rad (1.0 /deg), 감쇠 약 1.1, 최대 토크 2.0 을 쓴다.
로봇이 서 있지도 못하고 주저앉으면 강성부터 올려 본다.

---

## 6. 처음 돌릴 때 확인할 것

학습을 길게 돌리기 전에 짧게 창으로 띄워서 눈으로 본다.

```bash
python tasks/obstacle_rl/train.py --num_envs 8 --max_iterations 2
```

- [ ] 지형이 뜨고 칸마다 방지턱 / 구덩이 / 계단이 보이는가
- [ ] **파이프가 파이프 칸과 맵 1, 2 칸에만 장애물 앞에 놓이는가** (다른 칸에서는 로봇 뒤에 치워져 있어야 함).
      칸 종류를 계산하는 규칙을 Isaac Lab 소스와 대조하지 못했다. 엉뚱한 칸에 놓이면 `course.column_kinds` 를 확인
- [ ] 로봇이 바닥에 서는가 (뚫고 빠지거나 튀어 오르지 않는가)
- [ ] 다리 18개가 움직이고 팔은 접힌 자세로 가만히 있는가

### 자주 날 것 같은 에러

| 증상 | 볼 곳 |
|---|---|
| `rospider_float.usd 가 없다` 경고 | 1장 3번 (`--no-fix-base` 로 변환) |
| `Failed to create articulation` | 호스트 `nvidia-smi` 부터 (CLAUDE.md 함정 표). 그다음 USD 변환 다시 |
| 바디 / 조인트 이름을 못 찾는다 (`base_footprint`, `tibla_.*`, `femur_.*`) | `python tasks/sensor_fusion/02_urdf_to_usd.py --inspect` 로 실제 이름 확인 후 `robot_cfg.py` 의 `BASE_BODY`, `FOOT_BODIES` 수정 |
| 접촉 센서 관련 에러 | `robot_cfg.py` 의 `activate_contact_sensors=True` 확인, USD 다시 변환 |
| CUDA out of memory | `--num_envs` 를 줄인다 |
| `LLVM ERROR: out of memory` | VRAM 이 아니라 **시스템 RAM** 부족. 다른 프로그램을 끄거나 env 수를 줄인다 |
| `No module named 'isaaclab'` / `'rsl_rl'` | `bash /opt/isaaclab-steps/diag.sh` |

---

## 7. 알려진 한계

- **움직이는 파이프는 높이 스캔에 안 잡힌다.** 높이 스캔은 바닥 지형만 본다. 정책은 발이 닿아야 파이프를 안다.
- **높이 스캔은 시뮬에만 있는 정보다.** 실물 로봇에는 그런 센서가 없다. 실물로 옮기려면 뎁스 카메라 관측으로 바꾸는 단계(증류)가 필요하다.
- 파이프 관성은 꽉 찬 원통으로 계산돼 실제 강관보다 잘 구른다.
- 보상 가중치, PPO 설정, 다리 게인, 용접판 크기는 맞춰 보지 않은 출발값이다.
