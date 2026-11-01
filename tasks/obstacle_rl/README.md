# 장애물 넘기 강화학습 (`tasks/obstacle_rl/`)

RosPider 가 실물 장애물 코스(방지턱 / 구덩이 / 굴러가는 파이프 / 그레이팅 계단 → 작업대)를
넘도록 Isaac Lab 2.3.x + RSL-RL PPO 로 학습한다. `isaac-shell` 안, 레포 루트에서 실행한다.

```bash
python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base        # 1회: 몸이 고정되지 않은 USD
python tasks/obstacle_rl/test_course.py                           # 지형 검사 (Kit 불필요)
python tasks/obstacle_rl/train.py --headless --num_envs 64        # 학습 (--height_curriculum 옵션)
python tasks/obstacle_rl/play.py --checkpoint logs/rsl_rl/rospider_obstacle/<날짜>/model_3000.pt   # 실물 맵 1, 2 평가
```

| 파일 | 내용 |
|---|---|
| `course.py` | 실물 치수, 칸 배치, 메시 생성 (numpy 만) |
| `terrain_cfg.py` | Isaac Lab 지형 생성기 설정 (학습 맵 / 평가 맵) |
| `robot_cfg.py` | 떠 있는 베이스 USD, 다리 게인, 선 자세 |
| `env_cfg.py` | 관측 / 행동 / 명령 / 보상 / 종료 / 무작위화 |
| `mdp.py` | 굴러가는 파이프 배치, 전진 보상, 작업대 도착 판정 |
| `agents/rsl_rl_ppo_cfg.py` | PPO 설정 |
| `train.py`, `play.py` | 학습, 평가 |
| `test_course.py` | 지형 기하 검사 |

**상태 (2026-11-01)**: 지형 검사(`test_course.py`)만 통과. Isaac Lab 에서는 아직 돌리지 않았다.
보상 가중치, 다리 PD 게인, ㄴ자 용접판 크기는 맞춰 보지 않은 출발값이다.
