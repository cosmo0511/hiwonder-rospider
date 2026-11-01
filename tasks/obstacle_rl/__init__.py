"""RosPider 장애물 넘기 강화학습 과제 (Isaac Lab 2.3.x).

gym 등록만 여기서 한다. 실제 설정 모듈은 Kit 이 뜬 뒤에 import 된다(entry point 문자열).
train.py / play.py 는 이 패키지를 직접 import 하므로 Isaac Lab 의 train.py 를 쓸 필요는 없다.
"""

import gymnasium as gym

gym.register(
    id="RoSpider-Obstacle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoSpiderObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RoSpiderObstaclePPORunnerCfg",
    },
)
gym.register(
    id="RoSpider-Obstacle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoSpiderObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:RoSpiderObstaclePPORunnerCfg",
    },
)
