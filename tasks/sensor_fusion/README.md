# tasks/sensor_fusion — 뎁스 카메라 + IMU 융합

전체 설명은 **[`docs/sensor_fusion_manual.md`](../../docs/sensor_fusion_manual.md)** 에 있다.
여기는 커맨드 모음이다.

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

Kit 이 필요 없는 것부터 순서대로 통과시키면, 문제가 생겼을 때 어디가 원인인지 바로 안다.

## 순서

```bash
cd /workspace/rospider                      # 컨테이너 안 레포 경로

bash tasks/sensor_fusion/00_fetch_description.sh
pip install xacro
python tasks/sensor_fusion/01_xacro_to_urdf.py
python tasks/sensor_fusion/urdf_fk.py --joints joint2=0.85,joint3=-1.60,joint4=-1.26
python tasks/sensor_fusion/test_fusion.py          # "전부 통과" 를 보고 다음으로

python tasks/sensor_fusion/02_urdf_to_usd.py       # --view 로 눈 확인
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0  --tag level
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 20 --tag pitch20
python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 0 \
       --obstacle_height 0.02 --tag lowstep
```

결과는 `outputs/sensor_fusion/` 에 쌓인다.

## 자주 쓰는 플래그

```bash
--enable_cameras          # 카메라 센서가 있으면 필수. 헤드리스에서도 필요하다
--gui                     # 창을 띄운다 (호스트에서 xhost +local:root 먼저)
--dump_asset_info         # 바디/조인트 이름만 찍고 종료. prim 경로가 안 맞을 때
--pitch_deg 20            # 몸체를 앞으로 숙인다 (실험 조건)
--obstacle_height 0.02    # 넘어갈 수 있는 낮은 단차로 바꾼다
--dists 0.9,0.7,0.5,0.3   # 한 실행에서 장애물을 옮겨가며 측정
--floor_mode assumed_height   # 바닥 추정을 끄고 설치 높이 상수를 믿는다 (비교용)
--cam_width 96 --cam_height 72 --no_rgb   # VRAM 이 빠듯할 때
```
