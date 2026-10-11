"""WASD 키보드 조작. **Kit 이 뜬 뒤에 import 해야 한다** (carb/omni 를 쓴다).

Isaac Lab 에 `isaaclab.devices.Se2Keyboard` 가 이미 있지만 키 배치가 **방향키와
넘패드**다(v2.3.1 소스 확인). 게임처럼 WASD 로 몰고 싶어서 같은 carb 인터페이스를
써서 따로 만들었다. 구독/해제 방식은 그 구현을 그대로 따른다.

Se2Keyboard 와 다른 점이 하나 더 있다. 그쪽은 눌릴 때 명령에 **더하고** 떼일 때
**빼는데**, 창 포커스가 중간에 빠지면 KEY_RELEASE 를 못 받아 **명령이 눌린 채로
남는다**(로봇이 혼자 걸어간다). 여기서는 눌린 키의 **집합**을 들고 매번 거기서
명령을 다시 계산한다. 이벤트를 놓쳐도 다음 키 입력에서 복구되고, `release_all()`
한 번이면 확실히 멈춘다.

키 배치
-------
| 키 | 하는 일 |
|---|---|
| `W` / `S` | 전진 / 후진 |
| `A` / `D` | 좌측 / 우측으로 게걸음 (몸 방향은 그대로) |
| `Q` / `E` | 제자리 좌회전 / 우회전 |
| `SHIFT` (누른 채) | 속도 2배 (보행 생성기의 상한까지만) |
| `SPACE` | 즉시 정지 |

**창(뷰포트)에 포커스가 있어야 키가 들어온다.** 패널을 클릭하다가 안 먹으면
3D 뷰포트를 한 번 클릭하자.
"""

from __future__ import annotations

import weakref

import carb
import omni

# 키 -> (vx, vy, wz) 단위 방향. 몸 기준 +X 전방, +Y 좌측, +Z 상방(URDF 규약).
KEY_AXES: dict[str, tuple[float, float, float]] = {
    "W": (1.0, 0.0, 0.0),
    "S": (-1.0, 0.0, 0.0),
    "A": (0.0, 1.0, 0.0),
    "D": (0.0, -1.0, 0.0),
    "Q": (0.0, 0.0, 1.0),
    "E": (0.0, 0.0, -1.0),
}

HELP = """조작:  W/S 전/후   A/D 좌/우 게걸음   Q/E 좌/우 회전
       SHIFT 가속   SPACE 정지   (키가 안 먹으면 3D 뷰포트를 한 번 클릭)"""


class WasdTeleop:
    """눌린 키에서 (vx, vy, wz) 를 만들어 주는 장치.

    Example:
        teleop = WasdTeleop(v=0.12, w=0.8)
        vx, vy, wz = teleop.command()
    """

    def __init__(self, v: float = 0.12, w: float = 0.8, boost: float = 2.0) -> None:
        """
        Args:
            v: W/A/S/D 한 칸의 속도 [m/s].
            w: Q/E 한 칸의 각속도 [rad/s].
            boost: SHIFT 를 누르고 있을 때 곱할 배수. 실제 상한은 `GaitParams` 가
                다시 한 번 누르므로, 여기 값을 키워도 다리가 찢어지지는 않는다.
        """
        self.v = v
        self.w = w
        self.boost = boost
        self._held: set[str] = set()
        self._callbacks: dict[str, callable] = {}

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # weakref 로 넘겨야 이 객체가 지워질 때 구독도 같이 끊긴다 (Se2Keyboard 와 같은 이유).
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

    def __del__(self) -> None:
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:  # noqa: BLE001  종료 중에는 인터페이스가 이미 내려갔을 수 있다.
            pass
        self._sub = None

    # ------------------------------------------------------------------ 바깥 API
    def add_callback(self, key: str, func) -> None:
        """키 하나에 함수를 건다. 눌릴 때 한 번 불린다 (예: "R" -> 로봇 리셋)."""
        self._callbacks[key.upper()] = func

    def release_all(self) -> None:
        """눌린 키를 전부 잊는다. 즉시 정지."""
        self._held.clear()

    def command(self) -> tuple[float, float, float]:
        """지금 눌려 있는 키에 해당하는 (vx [m/s], vy [m/s], wz [rad/s])."""
        vx = vy = wz = 0.0
        for key in self._held:
            axes = KEY_AXES.get(key)
            if axes is None:
                continue
            vx += axes[0] * self.v
            vy += axes[1] * self.v
            wz += axes[2] * self.w
        if self.shift_held:
            vx, vy, wz = vx * self.boost, vy * self.boost, wz * self.boost
        return vx, vy, wz

    @property
    def shift_held(self) -> bool:
        return bool(self._held & {"LEFT_SHIFT", "RIGHT_SHIFT"})

    @property
    def pressed(self) -> str:
        """지금 눌린 조작키를 사람이 읽을 수 있게. 패널/터미널 표시용."""
        keys = [k for k in ("W", "A", "S", "D", "Q", "E") if k in self._held]
        if self.shift_held and keys:
            keys.append("SHIFT")
        return "+".join(keys) if keys else "-"

    # ------------------------------------------------------------------ 내부
    def _on_event(self, event, *args, **kwargs) -> bool:  # noqa: ARG002
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "SPACE":
                self.release_all()
            else:
                self._held.add(name)
            func = self._callbacks.get(name)
            if func is not None:
                func()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        # True 를 돌려줘야 이벤트가 계속 전달된다.
        return True
