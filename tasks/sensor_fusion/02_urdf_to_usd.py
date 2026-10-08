#!/usr/bin/env python3
"""URDF -> USD 변환. **Kit 을 띄우므로 isaac-shell 에서 돌려야 한다.**

과제 1번 목표(URDF 를 올려 USD 로 변환하고 씬에 세우기) 가 이 파일이다.

실행:
    # 헤드리스 (변환만)
    python tasks/sensor_fusion/02_urdf_to_usd.py

    # GUI 로 띄워 눈으로 확인 (URDF 작업은 눈으로 봐야 한다)
    python tasks/sensor_fusion/02_urdf_to_usd.py --view

변환이 끝나면 `assets/usd/rospider.usd` 가 생기고, 링크/조인트 이름을 찍어준다.
그 이름들이 `rospider_cfg.py` 의 prim 경로와 액추에이터 정규식에 그대로 들어간다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider URDF -> USD")
parser.add_argument(
    "--urdf",
    type=str,
    default="assets/rospider_description/rospider.urdf",
    help="01_xacro_to_urdf.py 가 만든 URDF",
)
parser.add_argument("--usd-dir", type=str, default="assets/usd", help="USD 출력 디렉터리")
parser.add_argument("--usd-name", type=str, default="rospider.usd")
parser.add_argument(
    "--merge-fixed-joints",
    action="store_true",
    default=True,
    help=(
        "fixed 조인트로 붙은 링크를 합친다(기본 켜짐). 바디 수가 51 -> 24 로 줄어 가볍다."
        " 단 imu_link/depth_cam_link 가 base_link 등으로 흡수돼 **별도 prim 이 사라진다.**"
        " 센서는 흡수된 부모 prim 에 offset 으로 붙인다(rospider_cfg.py 참고)."
    ),
)
parser.add_argument(
    "--no-merge-fixed-joints",
    dest="merge_fixed_joints",
    action="store_false",
    help="모든 링크를 개별 바디로 남긴다. imu_link 가 prim 으로 살아 있어 센서 부착이 직관적이지만,"
    " 질량/관성이 없는 더미 링크(depth_cam_frame 등) 때문에 PhysX 경고가 날 수 있다.",
)
parser.add_argument("--view", action="store_true", help="변환 후 GUI 로 띄워 확인")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.view:
    args_cli.headless = False

# ---- 여기서 Kit 이 뜬다. 이 줄 위에서는 omni.* 를 import 할 수 없다. ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def convert() -> Path:
    urdf_path = Path(args_cli.urdf).resolve()
    if not urdf_path.exists():
        raise SystemExit(
            f"{urdf_path} 가 없습니다.\n"
            "  bash tasks/sensor_fusion/00_fetch_description.sh\n"
            "  python tasks/sensor_fusion/01_xacro_to_urdf.py\n"
            "순서로 먼저 만드세요."
        )

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(Path(args_cli.usd_dir).resolve()),
        usd_file_name=args_cli.usd_name,
        force_usd_conversion=True,
        # 센서 융합 과제에서는 보행이 목표가 아니다. base 를 고정해 두면 다리 제어 없이도
        # 로봇이 서 있고, 자세(기울기)를 우리가 원하는 값으로 정확히 줄 수 있다.
        # 보행까지 가려면 False 로 바꾸고 다리 PD 게인부터 다시 잡아야 한다.
        fix_base=True,
        # URDF 의 루트는 base_footprint(더미) 다. base_link 를 루트로 지정해
        # merge 후에도 prim 이름이 base_link 로 남게 한다.
        root_link_name="base_link",
        merge_fixed_joints=args_cli.merge_fixed_joints,
        # STL 은 visual 과 collision 이 같은 파일이다. convex hull 로 단순화하면
        # 다리 접촉 계산이 가벼워진다. 발끝 모양이 중요해지면 convex_decomposition.
        collider_type="convex_hull",
        self_collision=False,
        # URDF 의 effort=1 N*m, velocity=1 rad/s 는 실기 서보(35 kg*cm = 약 3.4 N*m) 보다
        # 작게 적혀 있다. 여기서 드라이브 게인을 직접 주고, 한계값은 액추에이터 cfg 에서
        # 다시 덮는다(rospider_cfg.py).
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=20.0, damping=1.0),
        ),
        # 관성이 없는 더미 링크에 최소 밀도를 준다(더미 링크를 남길 때만 의미 있음).
        link_density=100.0,
    )

    converter = UrdfConverter(cfg)
    out = Path(converter.usd_path)
    print(f"[완료] USD: {out}")
    return out


def report(usd_path: Path) -> None:
    """변환 결과의 링크/조인트 이름을 찍는다. 센서 prim 경로를 정할 때 필요하다."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path))
    bodies, joints = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
        if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
            joints.append(prim.GetName())
    print(f"\n[리짓바디 {len(bodies)}개] {', '.join(sorted(bodies))}")
    print(f"\n[움직이는 조인트 {len(joints)}개] {', '.join(sorted(joints))}")
    print(
        "\n센서 prim 경로는 위 리짓바디 이름에서 고른다."
        " base_link 가 보이면 rospider_cfg.py 기본값이 맞다."
        " 안 보이면(예: base_footprint 로 합쳐졌다면) 그 이름으로 바꿔야 한다."
    )


def main() -> None:
    usd_path = convert()
    report(usd_path)

    if args_cli.view:
        print("\n[GUI] 씬에 세워 본다. 창을 닫으면 종료된다.")
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=3000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=3000.0))
        spawn_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
        spawn_cfg.func("/World/Robot", spawn_cfg, translation=(0.0, 0.0, 0.12))
        sim.reset()
        sim.set_camera_view(eye=[0.9, 0.9, 0.6], target=[0.0, 0.0, 0.15])
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
