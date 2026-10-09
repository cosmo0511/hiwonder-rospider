#!/usr/bin/env python3
"""뎁스 카메라 + IMU 융합 시뮬레이션 본체. **isaac-shell 에서 돌린다.**

하는 일:

1. 바닥/장애물/뒷벽이 있는 씬에 RosPider 를 세운다.
2. 팔 끝(link4)에 달린 뎁스 카메라와 몸체(base_link)의 IMU 를 매 스텝 읽는다.
3. 팔 각도에서 `R_cam<-imu` 를 매 스텝 계산하고(`extrinsics.py`), 뎁스와 IMU 를
   `fusion.py` 에 넘겨 하나의 판정을 받는다.
4. 조건마다 CSV + 4패널 PNG 를 남기고, `--record` 면 움직이는 GIF 까지 만든다.

두 가지 실행 모양
-----------------
**(A) 정지 스윕** (`--motion none`, 기본): 로봇을 고정하고 장애물만 옮겨가며
거리별 판정을 찍는다. 과제의 "조건을 바꾸어 결과를 비교" 용이다.

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
parser.add_argument("--floor_mode", choices=["estimated", "assumed_height"], default="estimated")
parser.add_argument("--out", default="outputs/sensor_fusion", help="결과(CSV/PNG/GIF/NPZ) 디렉터리")
parser.add_argument("--tag", default="run", help="결과 파일 이름에 붙일 꼬리표")
parser.add_argument("--gui", action="store_true", help="창을 띄운다")
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
from kit_exit import shutdown  # noqa: E402
from fusion import DECISION_NAMES, FusionParams, fuse  # noqa: E402
from rospider_cfg import ARM_OBSERVE_POSE, build_scene_cfg, pitch_to_quat  # noqa: E402


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
    ignore_impact = args_cli.ignore_impact or move_base

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

            panel = FusionPanel(args_cli.cam_width, args_cli.cam_height)
            print(
                "[GUI] 'RosPider | Depth + IMU Fusion' 창을 화면 왼쪽 위에 띄웠다."
                " 안 보이면 Isaac Sim 창을 옮겨 보거나, 아래 터미널 출력을 보면 된다."
            )
        except Exception as exc:
            import traceback as _tb

            print(f"[GUI] 패널을 못 만들었다(시뮬은 계속 돈다): {exc}")
            _tb.print_exc()
    params = FusionParams(floor_mode=args_cli.floor_mode)
    if ignore_impact:
        params.bump_acc_mps2 = float("inf")

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
        panel.update(
            result,
            d.cpu().numpy(),
            imu.projected_gravity_b[0].cpu().numpy(),
            imu.lin_acc_b[0].cpu().numpy(),
            cam_down,
        )

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
        csv_path = out_dir / f"{args_cli.tag}_log.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[완료] 로그 {csv_path} ({len(rows)} 행)")

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
                # 패널이 안 뜨는 환경도 있으므로 터미널에도 같은 값을 한 줄로 찍는다.
                # 같은 줄을 덮어써서 스크롤을 더럽히지 않는다.
                sys.stdout.write("\r" + result.summary() + "   ")
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
