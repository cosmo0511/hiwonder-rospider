"""설치 검증용 — Kit 을 띄우고 몇 스텝만 돌린 뒤 종료합니다.

튜토리얼의 create_empty.py 는 `while simulation_app.is_running(): sim.step()` 이라
사용자가 끄기 전까지 끝나지 않습니다. 검증 단계에서는 끝나는 게 중요하므로 별도로 둡니다.

    isaaclab -p _smoke_sim.py --headless            # 기본 60 스텝
    isaaclab -p _smoke_sim.py --headless --steps 10
    isaaclab -p _smoke_sim.py                       # GUI 로 보고 싶을 때
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab 설치 검증용 최소 시뮬레이션")
parser.add_argument("--steps", type=int, default=60, help="돌릴 물리 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Kit 앱 기동. 이 줄 전에는 omni.* 를 쓰는 모듈을 import 할 수 없습니다.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 여기서부터 omni 의존 모듈 import 가능 ----
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=0.01))
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.reset()

    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            raise RuntimeError(f"{i} 스텝에서 앱이 종료됐습니다")
        sim.step()

    print(f"[SMOKE-OK] {args_cli.steps} 스텝 완료")


if __name__ == "__main__":
    main()
    simulation_app.close()
