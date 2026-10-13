# tasks/construction_rl — 공사장 험지 보행 RL

**목표**: RosPider 가 공사장 같은 험지(자갈, 벽돌, 각재, 턱, 구덩이, 철판 틈)를
속도 명령대로 걷게 하는 RL 보행 정책. 위에서 VLA 가 속도 명령 (vx, vy, wz) 을 준다.

## 파일

| 파일 | Kit 필요? | 하는 일 | 상태 |
|---|---|---|---|
| `construction_terrain.py` | **예** | 공사장 험지 `TerrainGeneratorCfg` (11종, 10단계 커리큘럼) | 코드만 있음. [미검증] |
| `preview_terrain.py` | **예** | 지형만 생성해 높이 지도 PNG + `.ply` 저장 | [미검증] |

아직 RL env(관측, 보상, 액추에이터)는 없다. Isaac Lab 의
`isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` 를 바탕으로
RosPider 용으로 만들 예정이다.

## 실행

```bash
# isaac-shell 안, 레포 루트에서
python tasks/construction_rl/preview_terrain.py            # 기본
python tasks/construction_rl/preview_terrain.py --easy 0.5 # 높이를 절반으로
```

## 숫자를 왜 이렇게 잡았나

Isaac Lab 기본 험지는 ANYmal 기준(계단 5~23 cm, 블록 8 m)이다. RosPider 는 몸통 높이
0.12 m, 발 드는 높이 3.5 cm, 보폭 7 cm, 최고 속도 0.12 m/s 라 높이 상한을 5 cm,
블록을 3 m 로 줄였다. 5 cm 는 RL 이 지금 보행보다 다리를 더 들 수 있다고 보고 잡은
값이고, 실제로 넘을 수 있는지는 다리 기구학으로 아직 계산하지 않았다.
