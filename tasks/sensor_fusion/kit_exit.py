"""Kit 을 확실히 끝내는 헬퍼.

왜 필요한가: Isaac Sim 은 예외가 난 뒤 종료할 때 플러그인 언로드 단계에서 자주
멈춘다. 콘솔에 아래 같은 줄이 수십 줄 찍히고 그대로 매달린다.

    [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
    [Warning] [omni.physx.plugin] USD stage detach not called, holding a loose ptr to a stage!

이 상태에서는 **Ctrl+C 가 듣지 않는다.** 컨테이너를 밖에서 죽여야 하는데, 디버깅
중에 실행을 반복하는 상황에서는 이게 제일 큰 시간 낭비다.

그래서 종료를 두 겹으로 건다.
1. `simulation_app.close()` 를 정상적으로 시도한다.
2. 동시에 워치독 타이머를 걸어, 제한 시간 안에 안 끝나면 `os._exit()` 로 즉시
   프로세스를 끊는다. 파이썬 정리 절차를 건너뛰지만, 종료하는 마당이라 잃을 게 없다
   (GPU/메모리는 OS 가 회수한다).
"""

from __future__ import annotations

import os
import sys
import threading


def shutdown(simulation_app, code: int = 0, timeout: float = 20.0) -> None:
    """Kit 을 닫고 프로세스를 끝낸다. **이 함수는 돌아오지 않는다.**

    Args:
        simulation_app: `AppLauncher(...).app`
        code: 종료 코드.
        timeout: 이 시간(초) 안에 close() 가 안 끝나면 강제 종료한다.
    """

    def _force() -> None:
        sys.stderr.write(
            f"\n[kit_exit] Kit 종료가 {timeout:.0f}초 안에 안 끝나 강제로 끊는다."
            " (플러그인 언로드에서 매달리는 알려진 현상이다)\n"
        )
        sys.stderr.flush()
        os._exit(code)

    watchdog = threading.Timer(timeout, _force)
    watchdog.daemon = True
    watchdog.start()

    try:
        simulation_app.close()
    except Exception as exc:  # 종료 중 예외는 삼킨다. 어차피 끝낼 참이다.
        sys.stderr.write(f"[kit_exit] close() 중 예외(무시): {exc}\n")

    watchdog.cancel()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
