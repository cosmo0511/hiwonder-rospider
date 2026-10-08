#!/usr/bin/env python3
"""뎁스 카메라 + IMU 융합 시뮬레이션 본체. **isaac-shell 에서 돌린다.**

하는 일:

1. 바닥/장애물/뒷벽이 있는 씬에 RosPider 를 세운다.
2. 뎁스 카메라와 IMU 를 매 스텝 읽는다.
3. `fusion.py` 에 넘겨 하나의 판정(CLEAR/WARN/DANGER/STOP_*) 을 받는다.
4. 장애물 거리를 바꿔가며(한 번의 실행 안에서) CSV 로 기록하고,
   조건마다 센서 출력과 판정을 한 장에 담은 PNG 를 남긴다.

실행 예:

    # 조건 1: 수평
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras \
        --pitch_deg 0 --tag level

    # 조건 2: 20도 숙임 (같은 장애물, 같은 임계값)
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras \
        --pitch_deg 20 --tag pitch20

    # 조건 3: 넘어갈 수 있는 2 cm 단차
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras \
        --pitch_deg 0 --obstacle_height 0.02 --tag lowstep

    # 눈으로 보려면 (호스트에서 xhost +local:root 먼저)
    python tasks/sensor_fusion/run_fusion_demo.py --enable_cameras --gui --pitch_deg 20

`--enable_cameras` 는 **빼먹으면 안 된다.** 카메라 센서가 있는 씬은 헤드리스에서도
이 플래그가 필요하다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 뎁스+IMU 융합 데모")
parser.add_argument("--usd", default="assets/usd/rospider.usd", help="02_urdf_to_usd.py 결과")
parser.add_argument("--pitch_deg", type=float, default=0.0, help="몸체를 앞으로 숙이는 각도 [deg]")
parser.add_argument(
    "--dists",
    default="0.90,0.70,0.55,0.40,0.25",
    help="시험할 장애물 거리 목록 [m]. 카메라 기준 전방 수평거리. 한 번의 실행에서 순서대로 옮긴다.",
)
parser.add_argument("--obstacle_height", type=float, default=0.15, help="장애물 높이 [m]")
parser.add_argument("--settle_steps", type=int, default=40, help="거리를 옮긴 뒤 안정될 때까지 돌릴 스텝")
parser.add_argument("--sample_steps", type=int, default=10, help="조건마다 기록할 스텝 수")
parser.add_argument("--cam_width", type=int, default=160)
parser.add_argument("--cam_height", type=int, default=120)
parser.add_argument("--no_rgb", action="store_true", help="RGB 를 끈다. VRAM 이 빠듯하면.")
parser.add_argument("--floor_mode", choices=["estimated", "assumed_height"], default="estimated")
parser.add_argument("--out", default="outputs/sensor_fusion", help="결과(CSV/PNG/NPZ) 디렉터리")
parser.add_argument("--tag", default="run", help="결과 파일 이름에 붙일 꼬리표")
parser.add_argument("--gui", action="store_true", help="창을 띄운다")
parser.add_argument("--dump_asset_info", action="store_true", help="바디/조인트 이름만 찍고 종료")
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
from fusion import DECISION_NAMES, FusionParams, fuse  # noqa: E402
from rospider_cfg import CAM_HEIGHT_AT_LEVEL, build_scene_cfg  # noqa: E402


def save_figure(path: Path, rgb, depth, height, mask, title: str) -> None:
    """센서 출력 2개와 융합 중간 결과를 한 장에. 노션 노트의 '실험 결과' 용 그림이다."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[건너뜀] matplotlib 이 없어 PNG 를 못 만든다. `pip install matplotlib` 후 .npz 로 다시 그려도 된다.")
        return

    panels = []
    if rgb is not None:
        panels.append(("RGB (카메라)", rgb, None, None))
    finite = np.isfinite(depth)
    panels += [
        ("Depth [m]", np.where(finite, depth, np.nan), "viridis", None),
        ("IMU 보정 높이 [m]", height, "coolwarm", (-0.1, 0.4)),
        ("융합 장애물 마스크", mask.astype(float), "gray", (0, 1)),
    ]

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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  그림 저장: {path}")


def main() -> None:
    usd_path = Path(args_cli.usd)
    if not usd_path.exists():
        raise SystemExit(
            f"{usd_path} 가 없습니다. 먼저 변환하세요:\n"
            "  python tasks/sensor_fusion/02_urdf_to_usd.py"
        )

    dists = [float(x) for x in args_cli.dists.split(",")]
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim.set_camera_view(eye=[-0.7, -1.0, 0.7], target=[0.5, 0.0, 0.15])

    robot = scene["robot"]
    print(f"\n[바디 {len(robot.body_names)}개] {robot.body_names}")
    print(f"\n[조인트 {len(robot.joint_names)}개] {robot.joint_names}\n")
    if args_cli.dump_asset_info:
        return

    # 몸체를 숙이면 카메라가 낮아진다. assumed_height 모드를 쓸 때 쓰는 보정값.
    pitch = np.radians(args_cli.pitch_deg)
    cam_h = scene_cfg.robot.init_state.pos[2] + (
        -np.sin(pitch) * 0.13125 + np.cos(pitch) * 0.25086
    )
    params = FusionParams(floor_mode=args_cli.floor_mode, cam_height_m=float(cam_h))
    print(
        f"[설정] pitch={args_cli.pitch_deg}deg, 장애물 높이={args_cli.obstacle_height} m, "
        f"floor_mode={params.floor_mode}, 카메라 높이={cam_h:.3f} m "
        f"(수평일 때 {CAM_HEIGHT_AT_LEVEL:.3f} m)"
    )
    print(
        f"[임계값] DANGER<{params.d_danger_m} m, WARN<{params.d_warn_m} m, "
        f"기울기>{params.tilt_stop_deg}deg, 충격>{params.bump_acc_mps2} m/s^2\n"
    )

    # 조인트는 초기 자세를 그대로 목표로 준다. 팔은 단단한 게인으로 관측 자세를 붙잡는다.
    hold_target = robot.data.default_joint_pos.clone()

    csv_path = out_dir / f"{args_cli.tag}_log.csv"
    rows = []
    obstacle = scene["obstacle"]
    obstacle_state = obstacle.data.default_root_state.clone()
    cam_x_world = float(np.cos(pitch) * 0.13125 + np.sin(pitch) * 0.25086)

    for dist in dists:
        # 장애물을 이번 조건의 거리로 옮긴다 (+0.1 은 상자 중심까지).
        state = obstacle_state.clone()
        state[:, 0] = cam_x_world + dist + 0.1
        state[:, :3] += scene.env_origins
        obstacle.write_root_pose_to_sim(state[:, :7])
        obstacle.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))

        for step in range(args_cli.settle_steps + args_cli.sample_steps):
            robot.set_joint_position_target(hold_target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

            if step < args_cli.settle_steps:
                continue

            depth = scene["camera"].data.output["distance_to_image_plane"]
            intrinsics = scene["camera"].data.intrinsic_matrices
            imu = scene["imu"].data
            result = fuse(depth, intrinsics, imu.projected_gravity_b, imu.lin_acc_b, params)

            rows.append(
                {
                    "pitch_deg": args_cli.pitch_deg,
                    "obstacle_dist_m": dist,
                    "obstacle_height_m": args_cli.obstacle_height,
                    "step": step,
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
            )

        print(f"장애물 {dist:.2f} m -> {result.summary()}")

        # 마지막 샘플을 그림/원본으로 남긴다.
        rgb = None
        if "rgb" in scene["camera"].data.output:
            rgb = scene["camera"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        depth_np = depth[0].squeeze(-1).cpu().numpy() if depth.dim() == 4 else depth[0].cpu().numpy()
        height_np = result.height_map[0].cpu().numpy()
        mask_np = result.obstacle_mask[0].cpu().numpy()
        stem = f"{args_cli.tag}_d{dist:.2f}"
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            depth=depth_np,
            height=height_np,
            mask=mask_np,
            rgb=rgb if rgb is not None else np.zeros((1,)),
            intrinsics=intrinsics[0].cpu().numpy(),
            projected_gravity_b=imu.projected_gravity_b[0].cpu().numpy(),
            lin_acc_b=imu.lin_acc_b[0].cpu().numpy(),
        )
        save_figure(
            out_dir / f"{stem}.png",
            rgb,
            depth_np,
            height_np,
            mask_np,
            title=(
                f"pitch={args_cli.pitch_deg:.0f}deg  장애물 {dist:.2f} m (높이 {args_cli.obstacle_height} m)  "
                f"=>  융합 {DECISION_NAMES[int(result.decision[0])]} "
                f"(d={result.d_fused[0].item():.2f} m) / "
                f"뎁스단독 {DECISION_NAMES[int(result.decision_naive[0])]} / "
                f"고정ROI {DECISION_NAMES[int(result.decision_roi[0])]}"
            ),
        )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[완료] 로그 {csv_path} ({len(rows)} 행)")

    if args_cli.gui:
        print("[GUI] 창을 닫으면 종료된다.")
        while simulation_app.is_running():
            robot.set_joint_position_target(hold_target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
