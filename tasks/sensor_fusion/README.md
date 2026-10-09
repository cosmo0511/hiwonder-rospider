# tasks/sensor_fusion — 뎁스 카메라 + IMU 융합

전체 설명과 상세 실행 절차는
**[`docs/sensor_fusion_manual.md`](../../docs/sensor_fusion_manual.md)** 에 있다.
여기는 복붙용 요약이다.

## 파일

| 파일 | Kit 필요? | 하는 일 |
|---|---|---|
| `00_fetch_description.sh` | 아니오 | Hiwonder/ROSpider 에서 `rospider_description` 만 받아온다 |
| `01_xacro_to_urdf.py` | 아니오 | xacro → 단일 URDF. `pip install xacro` 만 있으면 된다 |
| `urdf_fk.py` | 아니오 | URDF 체인에서 카메라/IMU 포즈와 `R_cam<-imu` 를 뽑는다 |
| `fusion.py` | 아니오 | 융합 로직. 텐서만 받는다 |
| `test_fusion.py` | 아니오 | 합성 뎁스로 융합 7개 조건 검증 |
| `02_urdf_to_usd.py` | **예** | URDF → USD (`UrdfConverter`) |
| `rospider_cfg.py` | **예** | 로봇/센서/씬 설정 |
| `run_fusion_demo.py` | **예** | 본체. 조건별 CSV + 4패널 PNG 를 남긴다 |
| `run_all_conditions.sh` | **예** | 실험 조건 4개를 순서대로 돌린다 |

Kit 이 필요 없는 것부터 순서대로 통과시키면, 문제가 생겼을 때 어디가 원인인지 바로 안다.

## 복붙 순서

시뮬레이션은 **노트북(RTX 4060)의 도커 컨테이너 안**에서 돈다. 환경이 셋으로 갈리는
이유는 매뉴얼 5.0절, 전체 사양은 5.0.2절.

```bash
# ── 호스트 (노트북, Ubuntu 24.04) ───────────────────────
cd ~/hiwonder-rospider
git fetch origin && git checkout claude/great-tesla-700vl0 && git pull
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

# ── Kit 단계 (첫 실행은 셰이더 컴파일로 5~10분 멈춘 듯 보인다) ──
python tasks/sensor_fusion/02_urdf_to_usd.py                   # -> 리짓바디 목록에 base_link
python tasks/sensor_fusion/02_urdf_to_usd.py --view            # 눈으로 확인, 다리 자세 조정
bash tasks/sensor_fusion/run_all_conditions.sh                 # 조건 4개 (10분 내외)

# ── 스크린샷용 ──────────────────────────────────────────
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --pitch_deg 20
```

결과는 `outputs/sensor_fusion/` 에 쌓인다. 레포가 bind mount 라서 **호스트 레포의
같은 경로에 그대로 있다** — `docker cp` 가 필요 없다.

`(isaac_lab)` 이 활성화돼 있으면 `python` 과 `isaaclab -p` 는 같은 동작이다.
`isaac-shell` 안에서 ROS 를 source 하지 말 것.

## 조건 하나씩 돌리기

```bash
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0  --tag level
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 20 --tag pitch20
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0 \
       --obstacle_height 0.02 --tag lowstep
```

## 자주 쓰는 플래그

```bash
--enable_cameras          # 카메라 센서가 있으면 필수 (이 스크립트는 코드에서 강제로 켠다)
--gui                     # 창을 띄운다 (호스트에서 xhost +local:root 먼저)
--dump_asset_info         # 바디/조인트 이름만 찍고 종료. prim 경로가 안 맞을 때
--pitch_deg 20            # 몸체를 앞으로 숙인다 (실험 조건)
--obstacle_height 0.02    # 넘어갈 수 있는 낮은 단차로 바꾼다
--dists 0.9,0.7,0.5,0.3   # 한 실행에서 장애물을 옮겨가며 측정
--floor_mode assumed_height   # 바닥 추정을 끄고 설치 높이 상수를 믿는다 (비교용)
--no_rgb                  # VRAM 이 빠듯할 때 제일 먼저 끌 것
--cam_width 96 --cam_height 72
```
