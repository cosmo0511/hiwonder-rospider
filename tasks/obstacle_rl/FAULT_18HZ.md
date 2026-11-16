# 18 Hz 보행 + 다리 고장 (r18, f18)

실물 버스 서보 주기(약 18 Hz)에 맞춰 평지 보행을 다시 학습하고, 다리 하나가 고장 나도 걷게 한다.
코드: `servo_fault.py`, `experiments.py` 의 `r18` / `f18`, 장면 저장 `highlight_eval.py`.

**상태 (2026-11-16): 코드만 있음. Isaac 에서 한 번도 안 돌려 봤다.** 고장 뽑기 · 서보 읽기 흉내 · 18 Hz 설정은
Isaac 없이 가짜 env 로 숫자만 확인했다 (고장 76 % / limp 44 % · locked 37 % · weak 18 %, 읽기 지연 · 0.24 도 눈금, decimation 11).

## 왜 18 Hz 인가

| | 값 | 출처 |
|---|---|---|
| 서보 | HX-35H: UART 115200 baud, 위치 · 온도 · 전압만 돌려줌 (속도 · 전류 없음), 0 ~ 1000 = 0 ~ 240 도, 0.18 s/60 도 | Hiwonder 제품 페이지 |
| 버스 | 반이중, 서보 하나씩 묻고 답함. HX-35H 에는 여러 개 한 번에 읽는 명령이 없음 | Hiwonder 버스 서보 프로토콜 |
| 18 개 읽기 | 요청 6 B + 응답 8 B = 1.2 ms x 18 = 22 ms + 서보 응답 지연 | 계산 |
| 18 개 쓰기 | 10 B = 0.87 ms x 18 = 16 ms | 계산 |
| 합 | 38 ms + 응답 지연 · 중계 -> 약 55 ms = **약 18 Hz** | 계산, **실측 아님** |

ROSpider 위키에는 강화학습 · sim-to-real 내용이 없다 (MoveIt2 / Gazebo 뿐). 구조는 Jetson -> STM32F407 -> 버스 서보,
IMU(QMI8658) 는 STM32 보드에 있고 `/ros_robot_controller/imu_raw` 로 나온다.

실물에서 잴 것: 18 개 위치 읽기 1 회, 쓰기 1 회, 둘 다 1 회에 걸리는 시간. 다리 18 개가 버스 하나인지. 재서 주기가 다르면
`experiments._exp_r18` 의 `set_control_hz(env, 18.0)` 만 바꾸면 된다.

## r18: 3-m 을 18 Hz 로

연경 3-m 정책(`policy/blind_3m_model_16096.pt`)은 장애물 코스용이지만 평지 타일 30 % 에서 직진 · 정지 · 회전 명령도 배웠다.
입력(IMU + 관절 + 명령 + 박자, 최근 4 스텝)과 출력(CPG 조절 22 개)이 r18 과 같아서 **그대로 이어 받는다.** 처음부터 하면
평지 걸음이 잡히는 데만 1500 ~ 2000 회 걸렸다 (2-f). 이어 받는 쪽이 싸다. 다만 3-m 은 50 Hz 에서 배워서 최근 4 스텝이 0.08 s
였는데 18 Hz 에서는 0.22 s 다. 이어 받은 직후 걸음이 흐트러질 수 있고, 안 되면 `--resume` 없이 처음부터 돌려 비교한다.

바꾼 것 (나머지는 3-m 그대로)

| | 3-m | r18 |
|---|---|---|
| 지형 | 코스 (계단 · 파이프 ...) | 평지 |
| 정책 주기 | 50 Hz (0.005 s x 4) | 18.2 Hz (0.005 s x 11) |
| gamma / lambda | 0.99 / 0.95 | 0.973 / 0.868 (초 단위 시야 같게) |
| 관절 각 관측 | 시뮬 값 + 잡음 | 서보 읽기 흉내: 0 ~ 1 주기 지연, 0점 오차 ±0.03 rad, 0.24 도 눈금 |
| 관절 속도 관측 | 시뮬 값 | 읽은 각의 차분 (실물에서 그대로 계산) |
| 서보 목표 | 바로 바뀜 | 55 ms 동안 직선 보간 (서보 MOVE_TIME) |
| IMU | 각속도, 중력 방향, 방향 오차 | 같음 |

## f18: 다리 고장

에피소드마다 75 % 확률로 다리 하나 (6 개 중 무작위) 고장. 반은 다리 전체, 반은 관절 하나.

| 모드 | 흉내 | 비율 |
|---|---|---|
| limp (힘 빠짐) | 강성 0, 감쇠 0.05 ~ 0.5. 읽기값도 0 으로 가림 (죽은 서보) | 45 % |
| locked (굳음) | 선 자세 ±0.3 rad 무작위 각도에 고정 | 35 % |
| weak (약함) | 토크 15 ~ 50 % | 20 % |

- 정책에 고장 다리 one-hot 6 개를 준다 (`fault` 그룹). 실물에서는 사람이 아니까 (지금 LF) 그대로 넣으면 된다.
- tripod 보상(박자대로 딛기, 뜬 발, 미끄러짐, 발 높이)과 허벅지 충돌 벌점은 건강한 다리만 센다.
- 행동은 그대로 tripod CPG + 관절 보정 ±0.2 rad. **5 발 tripod 로 안 넘어지고 걸을 수 있는지는 미확인.** 안 되면 다음 단계는
  다리별 박자(위상) 조절을 행동에 넣는 것이다 (행동 크기가 바뀌어 처음부터 학습).

## 실행 (isaac-shell, 레포 루트)

```bash
# 1) r18: 3-m 이어 받기. 평가 명령 0.15 m/s (3-m 평지 속도 0.147)
TRAIN_EXTRA="--std_max 0.5 --bound_coef 0.1" bash tasks/obstacle_rl/run_exp.sh r18 1500 0.15 tasks/obstacle_rl/policy/blind_3m_model_16096.pt

# 2) f18: r18 마지막 체크포인트에 고장 입력 칸 붙여서 이어 받기 (actor +6, critic +60)
python tasks/obstacle_rl/expand_ckpt.py logs/rsl_rl/rospider_obstacle/<날짜>_expr18/model_XXXX.pt /tmp/r18_f.pt --actor_add 6 --critic_add 60
TRAIN_EXTRA="--std_max 0.5 --bound_coef 0.1" bash tasks/obstacle_rl/run_exp.sh f18 3000 0.12 /tmp/r18_f.pt
```

`run_exp.sh` 는 학습 뒤 자동으로 걸음새 평가, 영상, **넘어짐 / 잘 걷기 장면** 까지 만든다.
`run_exp.sh` 는 env 2048 개로 학습한다 (3-m 과 같음). 8 GB 에서 메모리가 모자라면 `run_exp.sh` 안의 `--num_envs` 를 줄인다.

## 장면 자동 저장 (`highlight_eval.py`)

로봇 16 마리를 30 초 걷게 하고, 마리마다 작은 카메라 (320 x 240) 로 최근 5 초를 들고 있다가

- 넘어지면 (시간 초과가 아닌 종료) 그 직전 5 초 -> `fall_00.mp4`, `fall_01.mp4` (처음 2 번)
- 끝날 때 안 넘어진 로봇 중 (전진 거리 - 2 x 옆 이동) 이 가장 큰 로봇의 마지막 5 초 -> `best_walk.mp4`
- 장면마다 가운데 프레임 png, `highlights.json` 에 고장 다리 · 모드 · 넘어진 이유 · 이동 거리

결과: `outputs/obstacle_rl/highlights/<체크포인트>_<실험>/`. f 실험이면 `run_exp.sh` 가 `@LF:limp` (왼쪽 앞 다리 힘 빠짐) 로 한 번 더 찍는다.

```bash
python tasks/obstacle_rl/highlight_eval.py --headless --enable_cameras --exp "f18@LF:limp" --checkpoint <ckpt>
```

평가 때 고장 고정: 실험 이름 뒤에 `@다리[.관절][:모드]` — `f18@LF`, `f18@LF:locked`, `f18@RM.tibla:weak`, `f18@none`.
`gait_eval.py`, `command_eval.py` 의 `--exp` 에도 똑같이 쓸 수 있다. `r18@LF` 로 고장을 안 배운 정책과 비교할 수 있다.

## 미검증 (사실로 쓰지 말 것)

- Isaac 실행 전부. `write_joint_stiffness_to_sim` 등 API 는 Isaac Lab v2.3.1 소스와 대조만 했다
- 18 Hz 는 계산값. 서보 응답 지연, 버스 구성 실측 필요
- 고장 서보가 실제로 limp / locked / weak 중 무엇에 가까운지. 실물 LF 를 손으로 돌려 보고 위치가 읽히는지 확인 필요
- 0점 오차 ±0.03 rad, 읽기 지연 0 ~ 1 주기는 가정값
- `highlight_eval.py` 의 카메라 16 대가 8 GB 에서 되는지, 넘어지는 순간 프레임이 리셋 전 장면인지
