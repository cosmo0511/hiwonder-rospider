#!/usr/bin/env python3
"""뎁스 카메라 + IMU 융합 시뮬레이션 본체. **isaac-shell 에서 돌린다.**

하는 일:

1. 바닥/장애물/뒷벽이 있는 씬에 RosPider 를 세운다.
2. 팔 끝(link4)에 달린 뎁스 카메라와 몸체(base_link)의 IMU 를 매 스텝 읽는다.
3. 팔 각도에서 `R_cam<-imu` 를 매 스텝 계산하고(`extrinsics.py`), 뎁스와 IMU 를
   `fusion.py` 에 넘겨 하나의 판정을 받는다.
4. 조건마다 CSV + 4패널 PNG 를 남기고, `--record` 면 움직이는 GIF 까지 만든다.

실행 모양 네 가지
-----------------
**(D) WASD 보행** (`--walk`): **이게 과제의 시나리오 그대로다.** 로봇이 걸어가면서
전방을 판단하고, DANGER 를 내면 **실제로 전진을 멈춘다.** 보행 자체가 몸통을
끄덕이게 하므로 `--pitch_deg` 로 손수 기울여 놓지 않아도 IMU 가 할 일이 생긴다.

    # 직접 몰기 (W/S 전후, A/D 게걸음, Q/E 회전, SHIFT 가속, SPACE 정지)
    python tasks/sensor_fusion/run_fusion_demo.py --walk --scene mixed

    # 사람 없이 (헤드리스). run_all_conditions.sh 가 쓰는 형태다.
    python tasks/sensor_fusion/run_fusion_demo.py --walk --auto_walk 0.10,0,0 \
        --walk_seconds 14 --tag walk --record

`--walk_mode kinematic`(기본)은 몸통을 명령 속도로 직접 옮긴다 — 항상 안정적이다.
`--walk_mode physics` 는 PhysX 접지로 진짜 걷는다(베이스 고정 안 한 USD 필요,
PD 게인이 [미검증]이라 넘어질 수 있다). 자세한 차이는 `--help` 참고.

**(A) 정지 스윕** (`--motion none`, 기본): 로봇을 고정하고 장애물만 옮겨가며
거리별 판정을 찍는다. 조건별 숫자를 깨끗하게 뽑을 때 쓴다.

**(B) 모션** (`--motion arm|rock|both`): 로봇을 움직이면서 센서 출력과 판정이
어떻게 변하는지 본다. `--record` 로 GIF 를 뽑으면 노션 노트의 "영상" 칸이 채워진다.

- `arm`  : 팔(joint3)을 흔들어 **카메라가 위아래를 훑는다.** 카메라 자세가 변하므로
           `R_cam<-imu` 도 매 스텝 바뀐다. 상수로 뒀다면 여기서 바닥 제거가 깨진다.
- `rock` : 몸체 pitch 를 흔든다. **IMU 값이 실제로 변한다.** base 가 고정되지 않은
           USD 가 필요하다(`02_urdf_to_usd.py --no-fix-base`).
- `both` : 둘 다.

실행 예:

    # 정지 스윕 (기본)
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --pitch_deg 20 --tag pitch20

    # 팔을 흔들며 GIF 녹화
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras \
        --motion arm --record --tag arm_sweep

    # 몸체를 흔들며 (float base USD 필요)
    python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras \
        --usd assets/usd/rospider_float.usd --motion rock --record --tag rock

    # 창을 띄워 눈으로 (호스트에서 xhost +local:root 먼저)
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --motion arm

`--enable_cameras` 는 코드에서 강제로 켜므로 안 쳐도 되지만, 직접 쓰는 스크립트에서는
**카메라 센서가 있으면 반드시 필요하다**(헤드리스에서도).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 뎁스+IMU 융합 데모")
parser.add_argument("--usd", default="assets/usd/rospider.usd", help="02_urdf_to_usd.py 결과")
parser.add_argument(
    "--urdf",
    default="assets/rospider_description/rospider.urdf",
    help="외부 파라미터를 매 스텝 계산하는 데 쓴다 (01_xacro_to_urdf.py 결과)",
)
parser.add_argument("--pitch_deg", type=float, default=0.0, help="몸체를 앞으로 숙이는 초기 각도 [deg]")
parser.add_argument(
    "--dists",
    default="0.90,0.70,0.55,0.40,0.25",
    help="정지 스윕에서 시험할 장애물 거리 [m]. 모션 모드에서는 첫 값만 쓴다.",
)
parser.add_argument("--obstacle_height", type=float, default=0.15, help="장애물 높이 [m]")
parser.add_argument(
    "--scene",
    choices=["box", "stairs", "mixed"],
    default="box",
    help="무엇을 놓을지. box=빨간 상자 하나(거리 스윕), stairs=계단 4단, "
    "mixed=넘어갈 수 있는 2cm 단차 + 계단. 시연에는 stairs/mixed 가 보기 좋다.",
)
parser.add_argument("--settle_steps", type=int, default=40, help="거리를 옮긴 뒤 안정될 때까지 돌릴 스텝")
parser.add_argument("--sample_steps", type=int, default=10, help="조건마다 기록할 스텝 수")

parser.add_argument(
    "--motion",
    choices=["none", "arm", "rock", "both"],
    default="none",
    help="로봇을 어떻게 움직일지. none 이면 정지 스윕.",
)
parser.add_argument("--motion_seconds", type=float, default=8.0, help="모션 모드에서 돌릴 시간 [s]")
parser.add_argument("--motion_period", type=float, default=4.0, help="흔드는 주기 [s]")
parser.add_argument("--arm_amp", type=float, default=0.45, help="팔 흔들기 진폭 [rad]")
parser.add_argument("--rock_amp_deg", type=float, default=18.0, help="몸체 pitch 흔들기 진폭 [deg]")
parser.add_argument(
    "--ignore_impact",
    action="store_true",
    help="충격(STOP_IMPACT) 판정을 끈다. rock 모드는 루트를 매 스텝 옮기므로 "
    "IMU 가속도가 물리적으로 의미 없는 값이 된다 — 그때 켜자(rock 에서는 자동으로 켜진다).",
)

parser.add_argument("--record", action="store_true", help="프레임을 모아 GIF 로 저장")
parser.add_argument("--record_every", type=int, default=4, help="몇 스텝마다 한 프레임 담을지")
parser.add_argument("--cam_width", type=int, default=160)
parser.add_argument("--cam_height", type=int, default=120)
parser.add_argument("--no_rgb", action="store_true", help="RGB 를 끈다. VRAM 이 빠듯하면.")
parser.add_argument(
    "--guessed_intrinsics",
    action="store_true",
    help="실기 캘리브레이션(camera_info.yaml, 화각 63.0도) 대신 예전 추정값(70도)을 쓴다. 비교용.",
)
parser.add_argument("--floor_mode", choices=["estimated", "assumed_height"], default="estimated")
parser.add_argument("--out", default="outputs/sensor_fusion", help="결과(CSV/PNG/GIF/NPZ) 디렉터리")
parser.add_argument("--tag", default="run", help="결과 파일 이름에 붙일 꼬리표")
parser.add_argument("--gui", action="store_true", help="창을 띄운다")
parser.add_argument(
    "--single_panel",
    action="store_true",
    help="센서 이미지를 창마다 따로 띄우지 않고 숫자 패널 하나에 다 쌓는다. "
    "기본은 창 4개로 나눠서 띄운다(스크롤 없이 한눈에 보려고).",
)
parser.add_argument(
    "--teleop",
    action="store_true",
    help="패널의 슬라이더로 몸체를 직접 기울인다. --gui 를 자동으로 켠다.\n"
    "**베이스가 고정되지 않은 USD 가 필요하다**:\n"
    "  python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base\n"
    "  python tasks/sensor_fusion/run_fusion_demo.py --teleop "
    "--usd assets/usd/rospider_float.usd --scene stairs\n"
    "fix_base=True 로 변환한 USD 는 루트가 월드에 용접돼 있어 포즈를 써도 안 움직인다.",
)
parser.add_argument(
    "--walk",
    action="store_true",
    help="**WASD 로 직접 몰면서** 융합 판정을 본다. 창을 자동으로 띄운다.\n"
    "  W/S 전후, A/D 게걸음, Q/E 회전, SHIFT 가속, SPACE 정지.\n"
    "--auto_walk 를 같이 주면 사람 없이 정해진 명령으로 걷는다(헤드리스 가능).",
)
parser.add_argument(
    "--walk_mode",
    choices=["kinematic", "physics"],
    default="kinematic",
    help="걷는 방식.\n"
    "  kinematic (기본): 다리는 보행 궤적대로 움직이고, **몸통은 명령 속도를 적분해**\n"
    "    직접 옮긴다. 접지 물리가 아니므로 항상 안정적으로 전진한다. 몸통 끄덕임은\n"
    "    gait.body_sway() 의 합성 모델이다. 융합 데모에는 이쪽이 안전하다.\n"
    "  physics: 조인트 목표만 주고 **PhysX 의 접지력으로 실제로 걷는다.** IMU 가\n"
    "    읽는 흔들림이 진짜다. 다만 PD 게인·마찰이 [미검증]이라 넘어질 수 있다.\n"
    "    **베이스가 고정되지 않은 USD 가 필요하다**: 02_urdf_to_usd.py --no-fix-base",
)
parser.add_argument(
    "--auto_walk",
    default=None,
    metavar="VX,VY,WZ",
    help="키보드 대신 고정 명령으로 걷는다. 예: --auto_walk 0.10,0,0 (전진 0.1 m/s).\n"
    "헤드리스로 돌아가므로 run_all_conditions.sh 가 이걸 쓴다.",
)
parser.add_argument("--walk_seconds", type=float, default=14.0,
                    help="--auto_walk 로 걸을 시간 [s]")
parser.add_argument(
    "--no_stop_on_danger",
    action="store_true",
    help="융합이 DANGER/STOP 을 내도 **멈추지 않는다.**\n"
    "기본은 멈춘다 — 판정이 그냥 숫자로 끝나지 않고 실제 행동으로 이어지는 것이\n"
    "이 과제가 요구하는 '최종 판단' 이기 때문이다. 멈춘 뒤에도 후진(S)과 회전(Q/E)은\n"
    "되므로 빠져나올 수 있다. 끄면 로봇이 장애물을 그대로 통과한다(비교용).",
)
parser.add_argument("--walk_speed", type=float, default=0.12, help="WASD 한 칸 속도 [m/s]")
parser.add_argument("--turn_speed", type=float, default=0.8, help="Q/E 한 칸 각속도 [rad/s]")
parser.add_argument("--gait_period", type=float, default=None, help="보행 주기 [s]")
parser.add_argument("--step_height", type=float, default=None, help="발 드는 높이 [m]")
parser.add_argument("--body_height", type=float, default=None, help="몸통 높이 [m]")
parser.add_argument(
    "--bump_acc",
    type=float,
    default=None,
    help="충격 정지 임계값 [m/s^2]. 안 주면 모드에 맞는 기본값을 쓴다(아래 설명 출력).",
)
parser.add_argument(
    "--imu_arrow",
    action="store_true",
    help="GUI 에 IMU 가속도 화살표 마커를 그린다. **기본은 꺼짐.** "
    "gravity_bias=(0,0,0) 이라 정지 상태에서는 가속도가 0 이고, 그러면 화살표가 "
    "길이 0 또는 엉뚱한 방향으로 공중에 떠 보인다. 콘솔에 "
    "'FabricManager::initializePointInstancer mismatched prototypes' 경고도 남긴다. "
    "몸체를 흔드는 --motion rock 에서만 의미가 있다.",
)
parser.add_argument("--dump_asset_info", action="store_true", help="바디/조인트 이름만 찍고 종료")
parser.add_argument(
    "--base_body",
    default="base_footprint",
    help="IMU 를 붙일 리짓바디 이름. `02_urdf_to_usd.py --inspect` 가 찍어 주는 목록에서 고른다.",
)
parser.add_argument("--arm_body", default="link4", help="카메라를 붙일 리짓바디 이름")
parser.add_argument(
    "--articulation_root",
    default=None,
    help="아티큘레이션 루트 prim 을 직접 지정한다(Robot prim 기준 상대 경로). "
    "'Failed to create articulation' 이 날 때 '' (Robot prim 자체) 나 '/base_link' 를 "
    "넣어 보면 USD 재변환 없이 빠르게 가릴 수 있다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 카메라 센서가 있는 씬은 이 플래그가 필수다. 깜빡하기 쉬우니 강제로 켠다.
args_cli.enable_cameras = True
if args_cli.teleop:
    args_cli.gui = True
if args_cli.walk and not args_cli.auto_walk:
    # 키보드로 몰려면 창이 떠 있어야 한다(carb 키보드 이벤트가 앱 창으로 온다).
    args_cli.gui = True
args_cli.headless = not args_cli.gui

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 여기서부터 omni 의존 모듈을 import 할 수 있다 ----
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extrinsics import ArmCameraExtrinsics  # noqa: E402
from gait import GaitParams, HexapodGait, advance_phase  # noqa: E402
from kit_exit import shutdown  # noqa: E402
from fusion import (  # noqa: E402
    DANGER,
    DECISION_NAMES,
    STOP_IMPACT,
    STOP_TILT,
    FusionParams,
    fuse,
)
from rospider_cfg import ARM_OBSERVE_POSE, build_scene_cfg, pitch_to_quat, rpy_to_quat  # noqa: E402


# --------------------------------------------------------------------------- 그림
# 그림 안 글자는 **영문으로만** 쓴다. 컨테이너(Ubuntu 22.04 최소 설치)에는 한글 폰트가
# 없어서 matplotlib 이 한글을 전부 □□□ 로 그린다. 노션 노트의 한글 설명은 그림 아래
# 캡션으로 적자. 굳이 그림 안에 한글을 넣고 싶으면 컨테이너에서
#   apt-get update && apt-get install -y fonts-nanum
#   python -c "import matplotlib; print(matplotlib.get_cachedir())"  # 이 캐시를 지우고
# 한 뒤 plt.rcParams["font.family"]="NanumGothic" 을 주면 된다.
def _panels(rgb, depth, height, mask):
    finite = np.isfinite(depth)
    out = []
    if rgb is not None:
        out.append(("RGB camera", rgb, None, None))
    out += [
        ("Depth [m]", np.where(finite, depth, np.nan), "viridis", None),
        ("Height above ground [m]\n(gravity-aligned, from IMU)", height, "coolwarm", (-0.1, 0.4)),
        ("Obstacle mask (fused)", mask.astype(float), "gray", (0, 1)),
    ]
    return out


def render_frame(rgb, depth, height, mask, title: str):
    """4패널 그림을 RGB 배열로 돌려준다. matplotlib 이 없으면 None."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None

    panels = _panels(rgb, depth, height, mask)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6))
    for ax, (name, img, cmap, lim) in zip(np.atleast_1d(axes), panels):
        kwargs = {"cmap": cmap} if cmap else {}
        if lim:
            kwargs["vmin"], kwargs["vmax"] = lim
        im = ax.imshow(img, **kwargs)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        if cmap:
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_png(path: Path, frame) -> None:
    if frame is None:
        print("[건너뜀] matplotlib 이 없어 PNG 를 못 만든다. `pip install matplotlib`")
        return
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("[건너뜀] PIL 이 없다.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)
    print(f"  그림 저장: {path}")


def walk_title(t: float, cmd, result, d_true: float) -> str:
    """보행 프레임의 그림 제목. 명령·자세·세 판정·참값을 한 줄에 담는다."""
    truth = "n/a" if d_true != d_true else f"{d_true:.2f} m"  # nan 검사
    return (
        f"t={t:4.1f}s   cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})   "
        f"body tilt {result.tilt_deg[0].item():4.1f}deg   true {truth}   =>   "
        f"FUSED {DECISION_NAMES[int(result.decision[0])]} "
        f"(d={result.d_fused[0].item():.2f} m)   |   "
        f"depth-only {DECISION_NAMES[int(result.decision_naive[0])]}   |   "
        f"fixed-ROI {DECISION_NAMES[int(result.decision_roi[0])]}"
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    """로그를 CSV 로. 조건마다 `--tag` 가 다르므로 파일이 서로 덮어쓰지 않는다."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[완료] 로그 {path} ({len(rows)} 행)")


def save_gif(path: Path, frames: list, fps: float = 8.0) -> None:
    frames = [f for f in frames if f is not None]
    if not frames:
        print("[건너뜀] 담긴 프레임이 없어 GIF 를 못 만든다.")
        return
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("[건너뜀] PIL 이 없어 GIF 를 못 만든다. `pip install pillow`")
        return
    images = [Image.fromarray(f) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"\n[완료] GIF {path} ({len(images)} 프레임)")


# --------------------------------------------------------------------------- 본체
def main() -> None:
    usd_path = Path(args_cli.usd)
    if not usd_path.exists():
        raise SystemExit(
            f"{usd_path} 가 없습니다. 먼저 변환하세요:\n  python tasks/sensor_fusion/02_urdf_to_usd.py"
        )
    urdf_path = Path(args_cli.urdf)
    if not urdf_path.exists():
        raise SystemExit(
            f"{urdf_path} 가 없습니다. 외부 파라미터 계산에 필요합니다:\n"
            "  python tasks/sensor_fusion/01_xacro_to_urdf.py"
        )

    dists = [float(x) for x in args_cli.dists.split(",")]
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    moving = args_cli.motion != "none"
    move_arm = args_cli.motion in ("arm", "both")
    move_base = args_cli.motion in ("rock", "both")
    walk_kinematic = args_cli.walk and args_cli.walk_mode == "kinematic"
    # 루트를 매 스텝 직접 옮기면 속도 차분으로 구하는 lin_acc_b 가 의미를 잃는다.
    # 자세(projected_gravity_b)는 정확하므로 바닥 제거는 그대로 동작한다.
    ignore_impact = (
        args_cli.ignore_impact or move_base or args_cli.teleop or walk_kinematic
    )
    if args_cli.walk and args_cli.walk_mode == "physics" and "float" not in usd_path.name:
        print(
            f"[경고] --walk_mode physics 인데 USD 가 {usd_path.name} 다. fix_base=True 로\n"
            "       변환한 USD 는 루트가 월드에 용접돼 있어 **제자리에서 다리만 젓는다.**\n"
            "       먼저:  python tasks/sensor_fusion/02_urdf_to_usd.py --no-fix-base\n"
            "       그 뒤:  --usd assets/usd/rospider_float.usd\n"
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))
    scene_cfg = build_scene_cfg(
        usd_path=str(usd_path.resolve()),
        pitch_deg=args_cli.pitch_deg,
        obstacle_dist=dists[0],
        obstacle_height=args_cli.obstacle_height,
        cam_width=args_cli.cam_width,
        cam_height=args_cli.cam_height,
        with_rgb=not args_cli.no_rgb,
        num_envs=1,
        debug_vis=args_cli.gui and args_cli.imu_arrow,
        use_real_intrinsics=not args_cli.guessed_intrinsics,
        scene=args_cli.scene,
        base_body=args_cli.base_body,
        arm_body=args_cli.arm_body,
        articulation_root=args_cli.articulation_root,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim.set_camera_view(eye=[-0.8, -1.1, 0.75], target=[0.5, 0.0, 0.2])

    robot = scene["robot"]
    print(f"\n[바디 {len(robot.body_names)}개] {robot.body_names}")
    print(f"\n[조인트 {len(robot.joint_names)}개] {robot.joint_names}\n")
    if args_cli.dump_asset_info:
        return

    # 외부 파라미터 계산기. 팔 각도를 넣으면 R_cam<-imu 를 돌려준다.
    ext = ArmCameraExtrinsics(urdf_path)

    # 창을 띄웠으면 오른쪽에 실시간 패널을 붙인다. 실패해도 시뮬은 계속 돈다.
    panel = None
    if args_cli.gui:
        try:
            from ui_panel import FusionPanel  # noqa: PLC0415

            panel = FusionPanel(
                args_cli.cam_width,
                args_cli.cam_height,
                enable_control=args_cli.teleop,
                separate_windows=not args_cli.single_panel,
            )
            print(
                "[GUI] 창을 띄웠다. 끌어서 옮기거나 Isaac Sim 패널에 도킹할 수 있다:\n"
                "        RosPider | Depth + IMU Fusion   판정 + 숫자"
                + ("" if args_cli.single_panel else
                   "\n        1. Depth [m]                    카메라 원본"
                   "\n        2. Height above ground          IMU 로 정렬한 높이맵"
                   "\n        3. Obstacle mask                최종 판단에 쓰인 픽셀"
                   "\n        4. d_fused vs d_roi             시계열 그래프")
            )
        except Exception as exc:
            import traceback as _tb

            print(f"[GUI] 패널을 못 만들었다(시뮬은 계속 돈다): {exc}")
            _tb.print_exc()
    params = FusionParams(floor_mode=args_cli.floor_mode)
    if ignore_impact:
        params.bump_acc_mps2 = float("inf")
    elif args_cli.walk:
        # 물리로 걸으면 **발이 땅에 닿을 때마다** 가속도가 튄다. 서 있을 때를 기준으로
        # 잡은 6 m/s^2 을 그대로 쓰면 보행 내내 STOP_IMPACT 라 거리 판정이 안 보인다.
        # 충돌 충격과 보행 발소리를 가르려면 원래는 저역통과가 필요하다(노트의 개선 방향).
        # 여기서는 임계값만 올려 두고, 그 사실을 출력에 남긴다.
        params.bump_acc_mps2 = 25.0
    if args_cli.bump_acc is not None:
        params.bump_acc_mps2 = args_cli.bump_acc
        ignore_impact = False

    print(
        f"[설정] motion={args_cli.motion}, pitch={args_cli.pitch_deg}deg, "
        f"장애물 높이={args_cli.obstacle_height} m, floor_mode={params.floor_mode}"
    )
    print(
        f"[임계값] DANGER<{params.d_danger_m} m, WARN<{params.d_warn_m} m, "
        f"기울기>{params.tilt_stop_deg}deg, "
        f"충격>{'끔' if ignore_impact else str(params.bump_acc_mps2) + ' m/s^2'}\n"
    )

    hold_target = robot.data.default_joint_pos.clone()
    joint_names = list(robot.joint_names)
    j3_index = joint_names.index("joint3") if "joint3" in joint_names else None
    j3_base = ARM_OBSERVE_POSE["joint3"]

    obstacle = scene["obstacle"]
    obstacle_state = obstacle.data.default_root_state.clone()
    pitch0 = math.radians(args_cli.pitch_deg)
    cam_x_world = float(math.cos(pitch0) * 0.13125 + math.sin(pitch0) * 0.25086)
    root_state0 = robot.data.default_root_state.clone()

    rows: list[dict] = []
    frames: list = []
    sim_dt = sim.get_physics_dt()

    def read_and_fuse():
        """센서를 읽고 외부 파라미터를 갱신해 융합한다."""
        depth = scene["camera"].data.output["distance_to_image_plane"]
        intrinsics = scene["camera"].data.intrinsic_matrices
        imu = scene["imu"].data
        # 팔 각도 -> R_cam<-imu. 실기의 '엔코더 + tf' 에 해당한다.
        q = ext.arm_joint_dict(joint_names, robot.data.joint_pos[0].cpu().numpy())
        r_cam_imu = ext.r_cam_imu(q)
        if params.floor_mode == "assumed_height":
            params.cam_height_m = float(
                robot.data.root_pos_w[0, 2].item() + ext.cam_height_in_base(q)
            )
        result = fuse(depth, intrinsics, imu.projected_gravity_b, imu.lin_acc_b, params, r_cam_imu)
        return depth, intrinsics, imu, result, q

    def update_panel(q) -> None:
        if panel is None:
            return
        d = depth[0].squeeze(-1) if depth.dim() == 4 else depth[0]
        cam_down = math.degrees(math.asin(max(-1.0, min(1.0, -float(ext.cam_pose_in_base(q)[1][2, 2])))))
        # imu_link 는 base 대비 yaw -90도라, 그대로 쓰면 roll/pitch 가 뒤바뀐다.
        # base 프레임으로 돌려서 넘긴다.
        g_base = ext.r_base_imu @ imu.projected_gravity_b[0].cpu().numpy()
        panel.update(result, d.cpu().numpy(), g_base, imu.lin_acc_b[0].cpu().numpy(), cam_down)

    def log_row(**extra) -> dict:
        row = {
            "motion": args_cli.motion,
            "pitch_deg": args_cli.pitch_deg,
            "obstacle_height_m": args_cli.obstacle_height,
            "tilt_deg": round(result.tilt_deg[0].item(), 3),
            "acc_mps2": round(result.bump_acc[0].item(), 3),
            "d_fused_m": round(result.d_fused[0].item(), 4),
            "d_naive_m": round(result.d_naive[0].item(), 4),
            "d_roi_m": round(result.d_roi[0].item(), 4),
            "obstacle_px": int(result.n_obstacle_px[0].item()),
            "decision_fused": DECISION_NAMES[int(result.decision[0])],
            "decision_naive": DECISION_NAMES[int(result.decision_naive[0])],
            "decision_roi": DECISION_NAMES[int(result.decision_roi[0])],
        }
        row.update(extra)
        return row

    def grab(title: str):
        rgb = None
        if "rgb" in scene["camera"].data.output:
            rgb = scene["camera"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        d = depth[0].squeeze(-1) if depth.dim() == 4 else depth[0]
        return (
            rgb,
            d.cpu().numpy(),
            result.height_map[0].cpu().numpy(),
            result.obstacle_mask[0].cpu().numpy(),
            title,
        )

    # 계단 씬에서는 빨간 상자를 멀리 치워 두었다(build_scene_cfg). 거리 스윕이 그걸
    # 다시 로봇 앞으로 끌어오면 계단과 상자가 겹쳐 보여 헷갈린다. 그래서 box 씬에서만
    # 상자를 옮기고, 계단 씬은 한 조건만 돌린다.
    move_obstacle = args_cli.scene == "box"
    if not move_obstacle:
        dists = dists[:1]

    if args_cli.walk:
        # ---------------- (D) WASD 보행 ----------------
        # 과제의 시나리오가 그대로 돌아가는 모드다. 로봇이 **걸으면서** 전방을 판단한다.
        # 몸통 끄덕임이 보행에서 저절로 나오므로, --pitch_deg 로 손수 기울여 놓지
        # 않아도 IMU 가 할 일이 생긴다.
        gp = GaitParams()
        if args_cli.gait_period is not None:
            gp.period_s = args_cli.gait_period
        if args_cli.step_height is not None:
            gp.step_height_m = args_cli.step_height
        if args_cli.body_height is not None:
            gp.body_height_m = args_cli.body_height
        gp.max_v_mps = max(gp.max_v_mps, args_cli.walk_speed)
        gp.max_w_radps = max(gp.max_w_radps, args_cli.turn_speed)
        gait = HexapodGait(urdf_path, gp)

        # 18개 다리 조인트를 한 번만 색인해 둔다. 매 스텝 이름으로 찾으면 비싸다.
        leg_joint_names = list(gait.stand_joint_targets().keys())
        missing = [nm for nm in leg_joint_names if nm not in joint_names]
        if missing:
            raise SystemExit(
                f"USD 에 다리 조인트가 없습니다: {missing}\n"
                "  python tasks/sensor_fusion/02_urdf_to_usd.py --inspect 로 실제 이름을 확인하세요."
            )
        leg_idx = torch.tensor(
            [joint_names.index(nm) for nm in leg_joint_names], device=robot.device
        )

        # 키보드. --auto_walk 면 사람 없이 고정 명령으로 돈다.
        teleop = None
        auto_cmd = None
        if args_cli.auto_walk:
            auto_cmd = tuple(float(v) for v in args_cli.auto_walk.split(","))
            if len(auto_cmd) != 3:
                raise SystemExit("--auto_walk 는 'VX,VY,WZ' 세 값입니다. 예: 0.10,0,0")
        else:
            try:
                from keyboard_wasd import HELP, WasdTeleop  # noqa: PLC0415

                teleop = WasdTeleop(v=args_cli.walk_speed, w=args_cli.turn_speed)
                print("\n" + HELP + "\n")
            except Exception as exc:  # noqa: BLE001
                print(f"[경고] 키보드를 못 잡았다({exc}). 제자리에 서 있는다. "
                      "--auto_walk 로 고정 명령을 주면 사람 없이 돈다.")
                auto_cmd = (0.0, 0.0, 0.0)

        print(
            f"[보행] mode={args_cli.walk_mode}, 주기 {gp.period_s}s, "
            f"발 드는 높이 {gp.step_height_m} m, 몸통 높이 {gp.body_height_m} m\n"
            f"       속도 상한 {gp.max_v_mps} m/s / {gp.max_w_radps} rad/s, "
            f"보폭 상한 {gp.max_stride_m} m"
        )
        if args_cli.walk_mode == "kinematic":
            print("       [주의] 운동학 모드다 — 몸통을 명령 속도로 직접 옮기고, 끄덕임은\n"
                  "              gait.body_sway() 의 합성 모델이다. 접지 물리가 아니다.\n"
                  "              진짜 접지로 걷게 하려면 --walk_mode physics (USD 는 float).")
        elif not ignore_impact:
            print(f"       [주의] 충격 임계값을 {params.bump_acc_mps2} m/s^2 로 올렸다. 보행은 발이\n"
                  "              닿을 때마다 가속도가 튀어서, 정지 기준(6)으로는 내내 STOP_IMPACT 다.")

        # 기준점. 운동학 모드에서 여기에 적분한 변위를 더한다.
        world_x = world_y = world_yaw = 0.0
        phase = 0.0
        obstacle_depth = 0.2  # build_scene_cfg 의 상자 깊이

        def ground_truth_distance() -> float:
            """카메라에서 장애물 앞면까지의 수평거리 [m]. d_fused 와 맞대 볼 정답이다.

            로봇이 회전하면 '전방'이 달라지므로, 루트 쿼터니언에서 뽑은 **진행 방향**에
            투영한다. 상자 씬이 아니면(계단/혼합) 상자를 멀리 치워 두므로 의미가 없다.
            """
            if not move_obstacle:
                return float("nan")
            cam = scene["camera"].data.pos_w[0]
            obs = obstacle.data.root_pos_w[0]
            qw, qx, qy, qz = (float(v) for v in robot.data.root_quat_w[0])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            dx = float(obs[0] - cam[0])
            dy = float(obs[1] - cam[1])
            return dx * math.cos(yaw) + dy * math.sin(yaw) - obstacle_depth / 2.0

        n_steps = (
            int(args_cli.walk_seconds / sim_dt) if auto_cmd is not None else None
        )
        step = 0
        blocked = False
        blocked_at: float | None = None
        png_every = max(int(2.0 / sim_dt), 1)  # 2초마다 한 장
        print()
        while simulation_app.is_running():
            if n_steps is not None and step >= n_steps:
                break
            t = step * sim_dt
            cmd = auto_cmd if teleop is None else teleop.command()
            # 인식 -> 행동. 막혔다고 판단하면 **앞으로 가는 성분만** 지운다.
            # 후진과 회전은 살려 둬야 빠져나올 수 있다.
            if blocked and not args_cli.no_stop_on_danger:
                cmd = (min(cmd[0], 0.0), cmd[1], cmd[2])
            phase = advance_phase(phase, sim_dt, gp.period_s)

            # 다리 18개는 보행 궤적대로, 나머지(팔/그리퍼)는 기본 자세 유지.
            target = hold_target.clone()
            angles = gait.joint_targets(phase, cmd)
            target[:, leg_idx] = torch.tensor(
                [angles[nm] for nm in leg_joint_names],
                device=target.device,
                dtype=target.dtype,
            )
            robot.set_joint_position_target(target)

            if walk_kinematic:
                # 명령 속도를 월드에서 적분한다. 몸 기준 (vx, vy) 를 yaw 로 돌려서 더한다.
                vx, vy, wz = gait.clip_cmd(cmd)
                world_yaw += wz * sim_dt
                world_x += (vx * math.cos(world_yaw) - vy * math.sin(world_yaw)) * sim_dt
                world_y += (vx * math.sin(world_yaw) + vy * math.cos(world_yaw)) * sim_dt
                dz, sway_pitch, sway_roll = gait.body_sway(phase, cmd)

                rs = root_state0.clone()
                rs[:, 0] += world_x
                rs[:, 1] += world_y
                rs[:, 2] += dz
                rs[:, 3:7] = torch.tensor(
                    rpy_to_quat(
                        math.degrees(sway_roll),
                        args_cli.pitch_deg + math.degrees(sway_pitch),
                        math.degrees(world_yaw),
                    ),
                    device=rs.device,
                    dtype=rs.dtype,
                ).unsqueeze(0)
                rs[:, :3] += scene.env_origins
                robot.write_root_pose_to_sim(rs[:, :7])
                robot.write_root_velocity_to_sim(torch.zeros_like(rs[:, 7:]))

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if step < 20:  # 초기 안정화
                step += 1
                continue

            # 매 물리 스텝(120 Hz)마다 융합하면 UI 갱신이 병목이다. 6 스텝마다 =
            # 20 Hz 로 판정한다. 실기 뎁스 카메라가 30 Hz 인 것과 같은 자릿수다.
            if step % 6 == 0:
                depth, intrinsics, imu, result, q = read_and_fuse()
                update_panel(q)
                d_true = ground_truth_distance()
                was_blocked = blocked
                blocked = int(result.decision[0]) in (DANGER, STOP_TILT, STOP_IMPACT)
                if blocked and not was_blocked and not args_cli.no_stop_on_danger:
                    blocked_at = d_true
                    print(
                        f"\n[정지] t={t:.1f}s  {DECISION_NAMES[int(result.decision[0])]} "
                        f"-> 전진을 막는다. 융합거리 {result.d_fused[0].item():.3f} m, "
                        f"참값 {d_true:.3f} m"
                    )
                rows.append(
                    log_row(
                        step=step,
                        t_s=round(t, 3),
                        cmd_vx=round(cmd[0], 4),
                        cmd_vy=round(cmd[1], 4),
                        cmd_wz=round(cmd[2], 4),
                        keys=("-" if teleop is None else teleop.pressed),
                        gait_phase=round(phase, 4),
                        swing="+".join(gait.swinging(phase)),
                        d_true_m=round(d_true, 4),
                        blocked=int(blocked),
                        obstacle_dist_m=dists[0],
                    )
                )
                if args_cli.record and step % args_cli.record_every == 0:
                    frames.append(render_frame(*grab(walk_title(t, cmd, result, d_true))))
                if step % png_every == 0:
                    save_png(
                        out_dir / f"{args_cli.tag}_t{t:04.1f}.png",
                        render_frame(*grab(walk_title(t, cmd, result, d_true))),
                    )
                if step % 60 == 0:
                    keys = "-" if teleop is None else teleop.pressed
                    sys.stdout.write(
                        f"\rt={t:5.1f}s [{keys:>12s}] "
                        f"tilt{result.tilt_deg[0].item():+5.1f} "
                        f"d_f{result.d_fused[0].item():6.3f} "
                        f"d_roi{result.d_roi[0].item():6.3f} "
                        f"참값{d_true:6.3f} "
                        f"{DECISION_NAMES[int(result.decision[0])]:<11s}"
                        f"/{DECISION_NAMES[int(result.decision_roi[0])]:<7s}"
                    )
                    sys.stdout.flush()
            step += 1

        print()
        if blocked_at is not None:
            print(
                f"[결과] 참거리 {blocked_at:.3f} m 에서 멈췄다 "
                f"(DANGER 임계값 {params.d_danger_m} m). "
                f"차이 {abs(blocked_at - params.d_danger_m) * 100:.1f} cm"
            )
        elif not args_cli.no_stop_on_danger:
            print("[결과] 끝까지 멈추지 않았다. 장애물까지 못 갔거나, 판정이 DANGER 에 "
                  "닿지 않았다 — --walk_seconds 를 늘리거나 로그의 d_true_m 을 보자.")
        if args_cli.record:
            save_gif(out_dir / f"{args_cli.tag}.gif", frames)
        if rows:
            save_png(out_dir / f"{args_cli.tag}_last.png",
                     render_frame(*grab("final frame")))
            write_rows(out_dir / f"{args_cli.tag}_log.csv", rows)
        return

    if args_cli.teleop:
        # ---------------- (C) 수동 조작 ----------------
        # 슬라이더가 가리키는 자세로 매 스텝 루트를 써 준다. fix_base=False 로 변환한
        # USD 여야 한다(고정 베이스는 월드에 용접돼 포즈를 무시한다).
        print("\n[teleop] 패널의 MANUAL CONTROL 슬라이더를 끌어 몸체를 기울여 보자.")
        print("         IMU pitch/roll 과 융합 판정이 따라 바뀐다. 창을 닫으면 끝난다.\n")
        step = 0
        while simulation_app.is_running():
            cmd_pitch, cmd_roll = (panel.read_control() if panel else (0.0, 0.0))
            rs = root_state0.clone()
            rs[:, 3:7] = torch.tensor(
                rpy_to_quat(cmd_roll, args_cli.pitch_deg + cmd_pitch),
                device=rs.device,
                dtype=rs.dtype,
            ).unsqueeze(0)
            rs[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(rs[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros_like(rs[:, 7:]))
            robot.set_joint_position_target(hold_target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            if step % 6 == 0:
                depth, intrinsics, imu, result, q = read_and_fuse()
                update_panel(q)
            step += 1
        return

    if not moving:
        # ---------------- (A) 정지 스윕 ----------------
        for dist in dists:
            if move_obstacle:
                state = obstacle_state.clone()
                state[:, 0] = cam_x_world + dist + 0.1
                state[:, :3] += scene.env_origins
                obstacle.write_root_pose_to_sim(state[:, :7])
                obstacle.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))

            for step in range(args_cli.settle_steps + args_cli.sample_steps):
                robot.set_joint_position_target(hold_target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                if step < args_cli.settle_steps:
                    continue
                depth, intrinsics, imu, result, q = read_and_fuse()
                rows.append(log_row(obstacle_dist_m=dist, step=step))
                update_panel(q)

            print(f"장애물 {dist:.2f} m -> {result.summary()}")
            title = (
                f"body pitch={args_cli.pitch_deg:.0f}deg   obstacle {dist:.2f} m   =>   "
                f"FUSED {DECISION_NAMES[int(result.decision[0])]} "
                f"(d={result.d_fused[0].item():.2f} m)   |   "
                f"depth-only {DECISION_NAMES[int(result.decision_naive[0])]}   |   "
                f"fixed-ROI {DECISION_NAMES[int(result.decision_roi[0])]}"
            )
            rgb, depth_np, height_np, mask_np, title = grab(title)
            frame = render_frame(rgb, depth_np, height_np, mask_np, title)
            save_png(out_dir / f"{args_cli.tag}_d{dist:.2f}.png", frame)
            frames.append(frame)
            np.savez_compressed(
                out_dir / f"{args_cli.tag}_d{dist:.2f}.npz",
                depth=depth_np, height=height_np, mask=mask_np,
                rgb=rgb if rgb is not None else np.zeros((1,)),
                intrinsics=intrinsics[0].cpu().numpy(),
                projected_gravity_b=imu.projected_gravity_b[0].cpu().numpy(),
                lin_acc_b=imu.lin_acc_b[0].cpu().numpy(),
                r_cam_imu=ext.r_cam_imu(q),
            )
    else:
        # ---------------- (B) 모션 ----------------
        dist = dists[0]
        if move_obstacle:
            state = obstacle_state.clone()
            state[:, 0] = cam_x_world + dist + 0.1
            state[:, :3] += scene.env_origins
            obstacle.write_root_pose_to_sim(state[:, :7])
            obstacle.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))

        n_steps = int(args_cli.motion_seconds / sim_dt)
        print(f"[모션] {args_cli.motion}, {args_cli.motion_seconds}s ({n_steps} 스텝), "
              f"주기 {args_cli.motion_period}s, 장애물 {dist:.2f} m\n")

        for step in range(n_steps):
            t = step * sim_dt
            phase = 2.0 * math.pi * t / args_cli.motion_period

            target = hold_target.clone()
            if move_arm and j3_index is not None:
                target[:, j3_index] = j3_base + args_cli.arm_amp * math.sin(phase)
            robot.set_joint_position_target(target)

            if move_base:
                # 루트를 직접 옮긴다. 자세(projected_gravity_b)는 정확하지만,
                # 속도 차분으로 구하는 lin_acc_b 는 의미가 없어지므로 충격 판정을 끈다.
                pitch = args_cli.pitch_deg + args_cli.rock_amp_deg * math.sin(phase)
                rs = root_state0.clone()
                rs[:, 3:7] = torch.tensor(
                    pitch_to_quat(pitch), device=rs.device, dtype=rs.dtype
                ).unsqueeze(0)
                rs[:, :3] += scene.env_origins
                robot.write_root_pose_to_sim(rs[:, :7])
                robot.write_root_velocity_to_sim(torch.zeros_like(rs[:, 7:]))

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if step < 20:  # 초기 안정화
                continue

            depth, intrinsics, imu, result, q = read_and_fuse()
            rows.append(log_row(obstacle_dist_m=dist, step=step, t_s=round(t, 3)))
            update_panel(q)

            if args_cli.record and step % args_cli.record_every == 0:
                cam_tilt = math.degrees(math.asin(-float(ext.cam_pose_in_base(q)[1][2, 2])))
                title = (
                    f"t={t:4.1f}s   body tilt {result.tilt_deg[0].item():5.1f}deg   "
                    f"camera down {cam_tilt:5.1f}deg   =>   "
                    f"FUSED {DECISION_NAMES[int(result.decision[0])]} "
                    f"(d={result.d_fused[0].item():.2f} m)   |   "
                    f"fixed-ROI {DECISION_NAMES[int(result.decision_roi[0])]}"
                )
                rgb, depth_np, height_np, mask_np, title = grab(title)
                frames.append(render_frame(rgb, depth_np, height_np, mask_np, title))

            if step % 120 == 0:
                print(f"t={t:4.1f}s  {result.summary()}")

        if args_cli.record:
            save_gif(out_dir / f"{args_cli.tag}.gif", frames)
        rgb, depth_np, height_np, mask_np, title = grab("final frame")
        save_png(out_dir / f"{args_cli.tag}_last.png",
                 render_frame(rgb, depth_np, height_np, mask_np, title))

    if rows:
        print()
        write_rows(out_dir / f"{args_cli.tag}_log.csv", rows)

    if not moving and args_cli.record:
        save_gif(out_dir / f"{args_cli.tag}.gif", frames, fps=1.5)

    if args_cli.gui:
        print("\n[GUI] 창을 닫으면 종료된다. 아래 한 줄이 계속 갱신된다(패널과 같은 값).")
        if args_cli.motion == "none":
            print("      --motion arm 을 주면 팔이 흔들려 센서 값이 변하는 걸 볼 수 있다.")
        print(
            "      d_fused=중력정렬 융합거리 / d_naive=뎁스단독 / d_roi=고정ROI,"
            " px=장애물 픽셀 수\n"
        )
        step = 0
        while simulation_app.is_running():
            t = step * sim_dt
            phase = 2.0 * math.pi * t / args_cli.motion_period
            target = hold_target.clone()
            if move_arm and j3_index is not None:
                target[:, j3_index] = j3_base + args_cli.arm_amp * math.sin(phase)
            robot.set_joint_position_target(target)
            if move_base:
                pitch = args_cli.pitch_deg + args_cli.rock_amp_deg * math.sin(phase)
                rs = root_state0.clone()
                rs[:, 3:7] = torch.tensor(
                    pitch_to_quat(pitch), device=rs.device, dtype=rs.dtype
                ).unsqueeze(0)
                rs[:, :3] += scene.env_origins
                robot.write_root_pose_to_sim(rs[:, :7])
                robot.write_root_velocity_to_sim(torch.zeros_like(rs[:, 7:]))
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            # 사람이 보는 구간이므로 몇 스텝에 한 번만 갱신한다(UI 갱신이 제일 비싸다).
            if step % 6 == 0:
                depth, intrinsics, imu, result, q = read_and_fuse()
                update_panel(q)
                # 패널이 안 뜨는 환경도 있으므로 터미널에도 짧게 한 줄 찍는다.
                # 터미널 폭(보통 80자)을 넘으면 \r 덮어쓰기가 안 되고 줄이 쌓이므로
                # summary() 전체가 아니라 핵심만 줄여서 쓴다.
                g_base = ext.r_base_imu @ imu.projected_gravity_b[0].cpu().numpy()
                pitch_deg = math.degrees(math.atan2(float(g_base[0]), -float(g_base[2])))
                d_f = result.d_fused[0].item()
                sys.stdout.write(
                    f"\rpitch{pitch_deg:+5.1f} d_f{d_f:6.3f} d_roi{result.d_roi[0].item():6.3f}"
                    f" px{int(result.n_obstacle_px[0].item()):5d} "
                    f"{DECISION_NAMES[int(result.decision[0])]:<11s}"
                    f"/{DECISION_NAMES[int(result.decision_roi[0])]:<7s}"
                )
                sys.stdout.flush()
            step += 1


if __name__ == "__main__":
    import traceback

    _code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        _code = 1
    # 예외가 나면 Kit 종료가 플러그인 언로드에서 매달린다. Ctrl+C 도 안 듣는다.
    # 워치독을 걸어 확실히 끝낸다(kit_exit.py 참고).
    shutdown(simulation_app, _code)
