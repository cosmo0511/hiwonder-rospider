"""Isaac Sim 창 안에 띄우는 센서 융합 패널. **Kit 이 뜬 뒤에만 import 할 수 있다.**

왼쪽 3D 뷰포트에는 로봇과 계단이 보이고, 이 패널에는 **두 센서가 지금 무엇을 보고
무슨 판정을 내렸는지**가 실시간으로 올라간다. 과제가 요구하는 "두 센서의 정보가 최종
판단에 어떻게 함께 쓰이는지" 를 한 화면에서 보여주는 부분이다.

패널 구성 (위에서부터)

1. **판정** — CLEAR / WARN / DANGER / STOP_TILT. 색으로도 구분한다.
2. **뎁스 카메라 쪽 숫자** — 융합 거리, 그리고 비교군 둘(뎁스 단독 / 고정 ROI).
   세 값이 **엇갈리는 순간**이 이 과제의 핵심 장면이다.
3. **IMU 쪽 숫자** — 몸체 기울기, IMU 프레임에서 본 중력 벡터, 카메라 하향각.
4. **이미지 3장** — Depth 원본 / IMU 로 보정한 높이맵 / 융합이 고른 장애물 픽셀.
   가운데 높이맵이 "IMU 가 뎁스에 무슨 일을 하는가" 를 그대로 보여준다.
5. **그래프** — 융합 거리와 고정 ROI 거리의 시계열.

omni.ui 는 Kit 안에서만 존재하므로 이 모듈은 `AppLauncher` 뒤에서 import 해야 한다.
"""

from __future__ import annotations

import numpy as np

from fusion import CLEAR, DANGER, DECISION_NAMES, STOP_IMPACT, STOP_TILT, WARN, FusionResult

# 판정별 색(RGBA, 0xAABBGGRR 형식이 아니라 omni.ui 가 쓰는 0xAABBGGRR 정수).
_DECISION_COLORS = {
    CLEAR: 0xFF4CAF50,       # 초록
    WARN: 0xFF00D7FF,        # 노랑
    DANGER: 0xFF3B30FF,      # 빨강
    STOP_TILT: 0xFFFF6BCB,   # 보라
    STOP_IMPACT: 0xFFFF6BCB,
}


def _to_rgb8(array: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    """float 배열을 컬러맵 입힌 uint8 RGB 로. ImagePlot 은 uint8 을 기대한다.

    `ImagePlot.update_image` 의 정규화 모드는 UI 체크박스로 켜는 것이라 기본은 꺼져
    있다. 그래서 여기서 직접 정규화하고 색을 입힌다. NaN/inf 는 검게 둔다.
    """
    finite = np.isfinite(array)
    norm = np.zeros_like(array, dtype=np.float32)
    span = max(vmax - vmin, 1e-6)
    norm[finite] = np.clip((array[finite] - vmin) / span, 0.0, 1.0)

    try:
        from matplotlib import colormaps  # noqa: PLC0415

        rgb = (colormaps[cmap_name](norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        # matplotlib 이 없으면 흑백으로. 기능은 그대로 돈다.
        gray = (norm * 255).astype(np.uint8)
        rgb = np.dstack([gray, gray, gray])

    rgb[~finite] = 0
    return np.ascontiguousarray(rgb)


class FusionPanel:
    """Isaac Sim 창 오른쪽에 붙는 실시간 패널."""

    def __init__(self, cam_width: int = 160, cam_height: int = 120, history: int = 300) -> None:
        import omni.ui as ui  # noqa: PLC0415

        from isaaclab.ui.widgets import ImagePlot, LinePlot  # noqa: PLC0415

        self._ui = ui
        blank = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)

        # dock_preference 를 쓰면 기존 탭 뒤로 숨어 버리는 일이 있었다(Window 메뉴에도
        # 안 뜬다 - 그 메뉴는 확장이 등록한 창만 보여 준다). 그래서 도킹을 쓰지 않고
        # 화면 왼쪽 위에 **떠 있는 창**으로 띄운다. 사용자가 끌어다 원하는 곳에 두면 된다.
        self.window = ui.Window("RosPider | Depth + IMU Fusion", width=430, height=880)
        self.window.visible = True
        self.window.position_x = 40
        self.window.position_y = 40

        self._labels: dict[str, object] = {}
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                # --- 1. 최종 판정 ---
                self._decision_label = ui.Label(
                    "DECISION: --",
                    height=46,
                    style={"font_size": 30, "color": 0xFFCCCCCC},
                )
                ui.Separator()

                # --- 2. 뎁스 카메라 ---
                ui.Label("DEPTH CAMERA (link4)", style={"font_size": 16, "color": 0xFF8CD2FF})
                for key, text in (
                    ("d_fused", "  d_fused  (gravity-aligned)"),
                    ("d_naive", "  d_naive  (depth only)"),
                    ("d_roi", "  d_roi    (fixed ROI)"),
                    ("px", "  obstacle pixels"),
                ):
                    self._labels[key] = self._row(text)
                ui.Separator()

                # --- 3. IMU ---
                ui.Label("IMU (base_link)", style={"font_size": 16, "color": 0xFF8CFFB0})
                for key, text in (
                    ("tilt", "  body tilt"),
                    ("gravity", "  gravity in IMU frame"),
                    ("cam_tilt", "  camera down-tilt"),
                    ("acc", "  linear accel"),
                ):
                    self._labels[key] = self._row(text)
                ui.Separator()

                # --- 4. 센서 이미지 ---
                ui.Label("WHAT THE FUSION SEES", style={"font_size": 16, "color": 0xFFCCCCCC})
                self.depth_plot = ImagePlot(
                    image=blank, label="Depth [m]", widget_height=150, show_min_max=False
                )
                self.height_plot = ImagePlot(
                    image=blank,
                    label="Height above ground (IMU-corrected)",
                    widget_height=150,
                    show_min_max=False,
                )
                self.mask_plot = ImagePlot(
                    image=blank, label="Obstacle mask (fused)", widget_height=150, show_min_max=False
                )

                # --- 5. 그래프 ---
                self.line_plot = None
                try:
                    self.line_plot = LinePlot(
                        y_data=[[0.0], [0.0]],
                        y_min=0.0,
                        y_max=1.5,
                        plot_height=130,
                        legends=["d_fused", "d_roi"],
                        max_datapoints=history,
                    )
                except Exception as exc:  # 그래프가 없어도 패널은 살아야 한다
                    print(f"[ui_panel] LinePlot 을 못 만들었다(무시하고 계속): {exc}")

    def _row(self, text: str):
        """'이름 ....... 값' 한 줄. 값 라벨을 돌려준다."""
        ui = self._ui
        with ui.HStack(height=20):
            ui.Label(text, width=230, style={"font_size": 14, "color": 0xFF999999})
            value = ui.Label("--", style={"font_size": 14, "color": 0xFFEEEEEE})
        return value

    @staticmethod
    def _fmt_dist(value: float) -> str:
        return "no obstacle" if not np.isfinite(value) else f"{value:.3f} m"

    def update(
        self,
        result: FusionResult,
        depth: np.ndarray,
        projected_gravity_b: np.ndarray,
        lin_acc_b: np.ndarray,
        cam_down_deg: float,
        env: int = 0,
    ) -> None:
        """한 스텝 분의 센서/판정을 패널에 밀어 넣는다."""
        code = int(result.decision[env])
        self._decision_label.text = f"DECISION: {DECISION_NAMES[code]}"
        self._decision_label.style = {
            "font_size": 30,
            "color": _DECISION_COLORS.get(code, 0xFFCCCCCC),
        }

        self._labels["d_fused"].text = self._fmt_dist(result.d_fused[env].item())
        self._labels["d_naive"].text = self._fmt_dist(result.d_naive[env].item())
        self._labels["d_roi"].text = self._fmt_dist(result.d_roi[env].item())
        self._labels["px"].text = str(int(result.n_obstacle_px[env].item()))

        self._labels["tilt"].text = f"{result.tilt_deg[env].item():.1f} deg"
        g = projected_gravity_b
        self._labels["gravity"].text = f"({g[0]:+.2f}, {g[1]:+.2f}, {g[2]:+.2f})"
        self._labels["cam_tilt"].text = f"{cam_down_deg:.1f} deg"
        a = lin_acc_b
        self._labels["acc"].text = f"{np.linalg.norm(a):.2f} m/s^2"

        self.depth_plot.update_image(_to_rgb8(depth, 0.2, 2.0, "viridis"))
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
