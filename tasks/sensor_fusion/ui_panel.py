"""Isaac Sim 창 안에 띄우는 센서 융합 패널. **Kit 이 뜬 뒤에만 import 할 수 있다.**

왼쪽 3D 뷰포트에는 로봇과 계단이 보이고, 이 창에는 **두 센서가 지금 무엇을 보고
무슨 판정을 내렸는지**가 실시간으로 올라간다. 과제가 요구하는 "두 센서의 정보가
최종 판단에 어떻게 함께 쓰이는지" 를 한 화면에서 보여주는 부분이다.

만들기 순서가 중요하다
----------------------
`ImagePlot` / `LinePlot` 은 Isaac Lab 의 위젯이라 버전이나 환경에 따라 생성이
실패할 수 있다. 그래서 **글자 라벨부터 먼저 만들고**, 이미지와 그래프는 각각
따로 try 로 감싼다. 하나가 실패해도 숫자는 계속 보인다. 예전에는 통째로 감싸서
위젯 하나가 죽으면 창이 아예 안 떴고, Window 메뉴에도 안 나와(직접 만든
`ui.Window` 는 그 메뉴에 등록되지 않는다) 찾을 방법이 없었다.
"""

from __future__ import annotations

import numpy as np

from fusion import CLEAR, DANGER, DECISION_NAMES, STOP_IMPACT, STOP_TILT, WARN, FusionResult

# 판정별 색 (omni.ui 는 0xAABBGGRR 정수를 쓴다).
_DECISION_COLORS = {
    CLEAR: 0xFF4CAF50,       # 초록
    WARN: 0xFF00D7FF,        # 노랑
    DANGER: 0xFF3B30FF,      # 빨강
    STOP_TILT: 0xFFFF6BCB,   # 보라
    STOP_IMPACT: 0xFFFF6BCB,
}

_HEAD = 0xFFCCCCCC
_DIM = 0xFF999999
_VAL = 0xFFEEEEEE
_CAM = 0xFF8CD2FF   # 뎁스 카메라 섹션 (파랑)
_IMU = 0xFF8CFFB0   # IMU 섹션 (초록)


def _to_rgb8(array: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    """float 배열을 컬러맵 입힌 uint8 RGB 로.

    `ImagePlot.update_image` 의 정규화 모드는 UI 체크박스라 기본이 꺼져 있다.
    그래서 여기서 직접 정규화하고 색을 입힌다. NaN/inf 는 검게 둔다.
    """
    finite = np.isfinite(array)
    norm = np.zeros_like(array, dtype=np.float32)
    span = max(vmax - vmin, 1e-6)
    norm[finite] = np.clip((array[finite] - vmin) / span, 0.0, 1.0)
    try:
        from matplotlib import colormaps  # noqa: PLC0415

        rgb = (colormaps[cmap_name](norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        rgb = np.dstack([gray, gray, gray])
    rgb[~finite] = 0
    return np.ascontiguousarray(rgb)


class FusionPanel:
    """Isaac Sim 창에 떠 있는 실시간 패널."""

    def __init__(self, cam_width: int = 160, cam_height: int = 120, history: int = 300) -> None:
        import omni.ui as ui  # noqa: PLC0415

        self._ui = ui
        self._labels: dict[str, object] = {}
        self.depth_plot = None
        self.height_plot = None
        self.mask_plot = None
        self.line_plot = None

        # 도킹하면 기존 탭 뒤로 숨는 일이 있었고, 직접 만든 ui.Window 는
        # Isaac Sim 의 Window 메뉴에도 안 올라와 찾을 방법이 없다.
        # 그래서 화면 왼쪽 위에 떠 있는 창으로 띄운다. 끌어서 옮기면 된다.
        self.window = ui.Window("RosPider | Depth + IMU Fusion", width=430, height=880)
        self.window.visible = True
        self.window.position_x = 40
        self.window.position_y = 60

        blank = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)

        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                # --- 최종 판정 ---
                self._decision_label = ui.Label(
                    "DECISION: --", height=46, style={"font_size": 30, "color": _HEAD}
                )
                ui.Separator()

                # --- 뎁스 카메라 ---
                ui.Label("DEPTH CAMERA  (on link4)", style={"font_size": 16, "color": _CAM})
                for key, text in (
                    ("d_fused", "  d_fused   gravity-aligned"),
                    ("d_naive", "  d_naive   depth only"),
                    ("d_roi", "  d_roi     fixed ROI"),
                    ("px", "  obstacle pixels"),
                    ("cam_tilt", "  camera down-tilt"),
                ):
                    self._labels[key] = self._row(text)
                ui.Separator()

                # --- IMU ---
                ui.Label("IMU  (on base_footprint)", style={"font_size": 16, "color": _IMU})
                self._labels["pitch"] = self._row("  pitch  (nose down +)", big=True)
                self._labels["roll"] = self._row("  roll   (right down +)", big=True)
                self._labels["tilt"] = self._row("  total tilt from vertical")
                self._labels["gravity"] = self._row("  gravity in IMU frame")
                self._labels["acc"] = self._row("  linear accel")
                ui.Separator()

                # --- 센서 이미지 (실패해도 위쪽 숫자는 남는다) ---
                ui.Label("WHAT THE FUSION SEES", style={"font_size": 16, "color": _HEAD})
                try:
                    from isaaclab.ui.widgets import ImagePlot  # noqa: PLC0415

                    self.depth_plot = ImagePlot(
                        image=blank, label="Depth [m]", widget_height=140, show_min_max=False
                    )
                    self.height_plot = ImagePlot(
                        image=blank,
                        label="Height above ground (IMU)",
                        widget_height=140,
                        show_min_max=False,
                    )
                    self.mask_plot = ImagePlot(
                        image=blank,
                        label="Obstacle mask (fused)",
                        widget_height=140,
                        show_min_max=False,
                    )
                except Exception as exc:
                    ui.Label(f"(이미지 위젯 생성 실패: {exc})", style={"color": _DIM})
                    print(f"[ui_panel] ImagePlot 실패(숫자 패널은 계속): {exc}")

                # --- 그래프 ---
                try:
                    from isaaclab.ui.widgets import LinePlot  # noqa: PLC0415

                    self.line_plot = LinePlot(
                        y_data=[[0.0], [0.0]],
                        y_min=0.0,
                        y_max=1.5,
                        plot_height=120,
                        legends=["d_fused", "d_roi"],
                        max_datapoints=history,
                    )
                except Exception as exc:
                    print(f"[ui_panel] LinePlot 실패(무시하고 계속): {exc}")

    def _row(self, text: str, big: bool = False):
        """'이름 ....... 값' 한 줄. 값 라벨을 돌려준다."""
        ui = self._ui
        size = 18 if big else 14
        with ui.HStack(height=24 if big else 20):
            ui.Label(text, width=235, style={"font_size": size, "color": _DIM})
            value = ui.Label("--", style={"font_size": size, "color": _VAL})
        return value

    @staticmethod
    def _fmt_dist(value: float) -> str:
        return "no obstacle" if not np.isfinite(value) else f"{value:.3f} m"

    def update(
        self,
        result: FusionResult,
        depth: np.ndarray,
        gravity_base: np.ndarray,
        lin_acc_b: np.ndarray,
        cam_down_deg: float,
        env: int = 0,
    ) -> None:
        """한 스텝 분의 센서/판정을 패널에 밀어 넣는다.

        Args:
            gravity_base: **base 프레임**에서 본 중력 단위벡터. IMU 프레임 값을
                그대로 쓰면 imu_link 의 yaw -90도 때문에 roll/pitch 가 뒤바뀐다.
        """
        code = int(result.decision[env])
        self._decision_label.text = f"DECISION: {DECISION_NAMES[code]}"
        self._decision_label.style = {
            "font_size": 30,
            "color": _DECISION_COLORS.get(code, _HEAD),
        }

        self._labels["d_fused"].text = self._fmt_dist(result.d_fused[env].item())
        self._labels["d_naive"].text = self._fmt_dist(result.d_naive[env].item())
        self._labels["d_roi"].text = self._fmt_dist(result.d_roi[env].item())
        self._labels["px"].text = str(int(result.n_obstacle_px[env].item()))
        self._labels["cam_tilt"].text = f"{cam_down_deg:.1f} deg"

        # 중력벡터에서 roll/pitch 를 뽑는다. 수평이면 (0, 0, -1) 이다.
        gx, gy, gz = (float(v) for v in gravity_base)
        pitch = np.degrees(np.arctan2(gx, -gz))
        roll = np.degrees(np.arctan2(-gy, -gz))
        self._labels["pitch"].text = f"{pitch:+6.1f} deg"
        self._labels["roll"].text = f"{roll:+6.1f} deg"
        self._labels["tilt"].text = f"{result.tilt_deg[env].item():.1f} deg"
        self._labels["gravity"].text = f"({gx:+.2f}, {gy:+.2f}, {gz:+.2f})"
        self._labels["acc"].text = f"{np.linalg.norm(lin_acc_b):.2f} m/s^2"

        if self.depth_plot is not None:
            self.depth_plot.update_image(_to_rgb8(depth, 0.1, 1.5, "viridis"))
            self.height_plot.update_image(
                _to_rgb8(result.height_map[env].cpu().numpy(), -0.05, 0.30, "coolwarm")
            )
            mask = result.obstacle_mask[env].cpu().numpy().astype(np.float32)
            self.mask_plot.update_image(_to_rgb8(mask, 0.0, 1.0, "gray"))

        if self.line_plot is not None:
            def clip(v: float) -> float:
                return 1.5 if not np.isfinite(v) else float(v)

            self.line_plot.add_datapoint(
                [clip(result.d_fused[env].item()), clip(result.d_roi[env].item())]
            )
