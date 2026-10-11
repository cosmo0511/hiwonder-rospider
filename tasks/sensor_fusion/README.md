# tasks/sensor_fusion — 걷는 로봇의 뎁스 카메라 + IMU 융합

**상황**: 사람이 RosPider 를 WASD 로 원격 조종해 걷게 한다. 로봇은 앞으로 몰리더라도
**부딪칠 것 앞에서는 스스로 멈추고**, 넘어갈 수 있는 2 cm 턱에서는 **멈추지 않아야**
한다. 카메라가 팔 끝에 달려 24도 아래를 보므로 화면 아래쪽은 늘 바닥이고, 6족 보행은
몸통이 계속 끄덕여서 "화면 위쪽만 본다" 같은 규칙이 통하지 않는다. IMU 의 **중력
방향**으로 뎁스 점의 높이를 구해 바닥을 기하학적으로 지운다.

전체 설명과 상세 실행 절차는
**[`docs/sensor_fusion_manual.md`](../../docs/sensor_fusion_manual.md)** 에 있다.
여기는 복붙용 요약이다.

## 파일

| 파일 | Kit 필요? | 하는 일 |
|---|---|---|
| `00_fetch_description.sh` | 아니오 | Hiwonder/ROSpider 에서 `rospider_description` 만 받아온다 |
| `01_xacro_to_urdf.py` | 아니오 | xacro → 단일 URDF. `pip install xacro` 만 있으면 된다 |
| `urdf_fk.py` | 아니오 | URDF 체인에서 카메라/IMU 포즈와 `R_cam<-imu` 를 뽑는다 |
| `extrinsics.py` | 아니오 | 팔 각도 → `R_cam<-imu` 를 매 스텝 계산 |
| `fusion.py` | 아니오 | 융합 로직. 텐서만 받는다 |
| `gait.py` | 아니오 | 삼각보행 + 다리 IK. 상수를 URDF 에서 뽑는다 |
| `test_fusion.py` | 아니오 | 합성 뎁스로 융합 11개 조건 검증 (K = 보행 중 판정) |
| `test_gait.py` | 아니오 | 보행 기구학을 URDF 와 대조 검증 |
| `keyboard_wasd.py` | **예** | WASD 키 입력 → (vx, vy, wz) |
| `02_urdf_to_usd.py` | **예** | URDF → USD (`UrdfConverter`) |
| `rospider_cfg.py` | **예** | 로봇/센서/씬 설정 |
| `ui_panel.py` | **예** | GUI 실시간 패널 |
| `run_fusion_demo.py` | **예** | 본체. 조건별 CSV + 4패널 PNG + GIF |
| `run_all_conditions.sh` | **예** | 실험 조건 9개를 돌리고 `summary.csv` 로 묶는다 |

Kit 이 필요 없는 것부터 순서대로 통과시키면, 문제가 생겼을 때 어디가 원인인지 바로 안다.

## 복붙 순서

시뮬레이션은 **노트북(RTX 4060)의 도커 컨테이너 안**에서 돈다. 환경이 셋으로 갈리는
이유는 매뉴얼 5.0절, 전체 사양은 5.0.2절.

```bash
# ── 호스트 (노트북, Ubuntu 24.04) ───────────────────────
cd ~/hiwonder-rospider
git fetch origin && git checkout main && git pull
xhost +local:root                                              # GUI 쓸 거면
docker compose -f docker/isaaclab/docker-compose.yml run --rm base

# ── 컨테이너 (매번 이 두 줄) ─────────────────────────────
source /opt/isaaclab-scripts/isaac-env.sh                      # (isaac_lab) 이 붙어야 한다
cd /workspace/rospider

# ── 최초 1회 ────────────────────────────────────────────
pip install xacro matplotlib

# ── Kit 없이 되는 단계 ──────────────────────────────────
bash tasks/sensor_fusion/00_fetch_description.sh               # -> 메시 36개
python tasks/sensor_fusion/01_xacro_to_urdf.py                 # -> revolute 29개
python tasks/sensor_fusion/urdf_fk.py --joints joint2=0.85,joint3=-1.60,joint4=-1.26
python tasks/sensor_fusion/test_fusion.py                      # -> "전부 통과"
python tasks/sensor_fusion/test_gait.py                        # -> "전부 통과"

# ── Kit 단계 (첫 실행은 셰이더 컴파일로 5~10분 멈춘 듯 보인다) ──
python tasks/sensor_fusion/02_urdf_to_usd.py                   # -> 리짓바디 목록에 base_link
python tasks/sensor_fusion/02_urdf_to_usd.py --view            # 눈으로 확인, 다리 자세 조정
bash tasks/sensor_fusion/run_all_conditions.sh                 # 조건 9개 (15분 내외)

# ── 직접 몰아 보기 (스크린샷·영상용) ─────────────────────
python tasks/sensor_fusion/run_fusion_demo.py --walk --scene mixed
```

결과는 `outputs/sensor_fusion/` 에 쌓인다. 레포가 bind mount 라서 **호스트 레포의
같은 경로에 그대로 있다** — `docker cp` 가 필요 없다.

`(isaac_lab)` 이 활성화돼 있으면 `python` 과 `isaaclab -p` 는 같은 동작이다.
`isaac-shell` 안에서 ROS 를 source 하지 말 것.

## WASD 조작

**창이 뜨면 3D 뷰포트를 한 번 클릭해 포커스를 준 뒤:**

| 키 | 하는 일 |
|---|---|
| `W` / `S` | 전진 / 후진 |
| `A` / `D` | 좌/우 게걸음 (몸 방향 유지) |
| `Q` / `E` | 제자리 좌/우 회전 |
| `SHIFT` | 누른 채로 가속 |
| `SPACE` | 즉시 정지 |

융합이 `DANGER`/`STOP_*` 를 내면 **전진이 막힌다.** 후진과 회전은 살아 있으므로
빠져나올 수 있다. 막는 걸 끄려면 `--no_stop_on_danger`.

## 조건 하나씩 돌리기

```bash
# 보행 (헤드리스, 결과 파일만)
python tasks/sensor_fusion/run_fusion_demo.py --walk --auto_walk 0.10,0,0 \
       --walk_seconds 16 --tag walk --record
python tasks/sensor_fusion/run_fusion_demo.py --walk --auto_walk 0.10,0,0 \
       --walk_seconds 16 --tag walk_nostop --no_stop_on_danger

# 정지 (숫자를 깨끗하게)
python tasks/sensor_fusion/run_fusion_demo.py --pitch_deg 0  --tag level
python tasks/sensor_fusion/run_fusion_demo.py --pitch_deg 20 --tag pitch20
python tasks/sensor_fusion/run_fusion_demo.py --pitch_deg 0 \
       --obstacle_height 0.02 --tag lowstep
```

> **`--tag` 를 꼭 바꾸자.** 로그가 `{tag}_log.csv` 로 저장되므로, 같은 tag 로 두 번
> 돌리면 앞 결과가 덮어써진다. 실제로 한 번 날렸다.

## 자주 쓰는 플래그

```bash
--walk                    # WASD 보행. 창을 자동으로 켠다
--auto_walk 0.10,0,0      # 키보드 대신 고정 명령 (헤드리스)
--walk_mode physics       # 진짜 접지로 걷기 (float USD 필요, 넘어질 수 있다)
--no_stop_on_danger       # DANGER 여도 안 멈춤 (비교군)
--scene mixed             # 2cm 턱 + 계단 + 상자. 시연에 제일 좋다
--record                  # GIF 녹화
--gui                     # 창을 띄운다 (호스트에서 xhost +local:root 먼저)
--dump_asset_info         # 바디/조인트 이름만 찍고 종료. prim 경로가 안 맞을 때
--pitch_deg 20            # 몸체를 앞으로 숙인다 (정지 실험 조건)
--obstacle_height 0.02    # 넘어갈 수 있는 낮은 단차로 바꾼다
--dists 0.9,0.7,0.5,0.3   # 한 실행에서 장애물을 옮겨가며 측정
--floor_mode assumed_height   # 바닥 추정을 끄고 설치 높이 상수를 믿는다 (의도된 실패)
--no_rgb                  # VRAM 이 빠듯할 때 제일 먼저 끌 것
--cam_width 96 --cam_height 72
```
