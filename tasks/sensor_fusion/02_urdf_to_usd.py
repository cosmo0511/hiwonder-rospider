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
parser.add_argument(
    "--root-link",
    default="base_link",
    help="어떤 링크를 아티큘레이션 루트로 삼을지. 빈 문자열('')이면 URDF 의 자연스러운 "
    "루트(base_footprint)를 쓴다. "
    "'Failed to create articulation' 가 나면 여기부터 '' 로 바꿔 보자.",
)
parser.add_argument(
    "--fix-base",
    dest="fix_base",
    action="store_true",
    default=True,
    help="base 를 월드에 고정한다(기본). 인식 과제라 보행이 필요 없고, 자세를 정확히 줄 수 있다.",
)
parser.add_argument(
    "--no-fix-base",
    dest="fix_base",
    action="store_false",
    help="base 를 띄운다. 파일 이름도 rospider_float.usd 로 바뀐다.",
)
parser.add_argument("--view", action="store_true", help="변환 후 GUI 로 띄워 확인")
parser.add_argument(
    "--inspect",
    action="store_true",
    help="변환하지 않고 **이미 있는 USD 를 열어** 구조만 찍는다. 아티큘레이션 루트가 "
    "어디에 붙어 있는지, 바디/조인트 이름이 무엇인지 본다. 물리를 안 돌리므로 "
    "'Failed to create articulation' 상황에서도 안전하게 쓸 수 있다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# AppLauncher 의 기본값은 **GUI 켜짐**이다(`HEADLESS` 환경변수 기본 0, `--headless` 는
# store_true 라 기본 False). 변환만 할 때 창이 뜨면 VRAM 을 3~4 GB 먼저 먹고,
# X11 이 안 잡힌 환경에서는 그대로 실패한다. 그래서 기본을 헤드리스로 뒤집고
# `--view` 일 때만 창을 띄운다.
args_cli.headless = not args_cli.view

# --inspect 는 USD 파일만 읽는다. pxr(USD 라이브러리)은 Kit 없이도 import 되므로
# **앱을 띄우지 않는다.** 덕분에 몇 초 만에 끝나고, Kit 종료가 매달리는 문제도 없다.
if args_cli.inspect:
    import sys

    from pxr import Usd, UsdPhysics  # noqa: E402

    def _usd_name_inspect() -> str:
        if args_cli.usd_name != "rospider.usd":
            return args_cli.usd_name
        return "rospider.usd" if args_cli.fix_base else "rospider_float.usd"

    target = Path(args_cli.usd_dir) / _usd_name_inspect()
    if not target.exists():
        sys.exit(f"{target} 가 없습니다. 먼저 --inspect 없이 돌려 변환하세요.")

    stage = Usd.Stage.Open(str(target))
    bodies, joints, fixed, roots = [], [], [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
        if prim.IsA(UsdPhysics.Joint):
            (fixed if prim.IsA(UsdPhysics.FixedJoint) else joints).append(prim.GetName())

    default_prim = stage.GetDefaultPrim()
    print(f"[검사] {target}")
    print(f"\n[기본 prim] {default_prim.GetPath() if default_prim else '(없음)'}")
    print(f"\n[아티큘레이션 루트 {len(roots)}개] {roots or '(없음!)'}")
    if len(roots) != 1:
        print(
            "  ** Isaac Lab 은 ArticulationRootAPI 가 붙은 prim 이 정확히 하나여야 한다. **\n"
            "     0개면 변환이 잘못된 것이고, 2개 이상이면 어느 쪽을 쓸지 몰라 실패한다."
        )
    print(f"\n[리짓바디 {len(bodies)}개] {', '.join(sorted(bodies))}")
    print(f"\n[움직이는 조인트 {len(joints)}개] {', '.join(sorted(joints))}")
    print(f"\n[고정 조인트 {len(fixed)}개] {', '.join(sorted(fixed))}")
    print(
        "\n센서 prim 경로는 위 리짓바디 이름에서 고른다."
        " base_link 와 link4 가 보이면 rospider_cfg.py 기본값이 맞다."
        " 다르면 run_fusion_demo.py 에 --base_body / --arm_body 로 넘기면 된다."
    )
    sys.exit(0)

# ---- 여기서 Kit 이 뜬다. 이 줄 위에서는 omni.* 를 import 할 수 없다. ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from kit_exit import shutdown  # noqa: E402


def _usd_name() -> str:
    """--no-fix-base 로 만든 것은 파일을 따로 둔다. 둘을 섞으면 헷갈린다."""
    if args_cli.usd_name != "rospider.usd":
        return args_cli.usd_name
    return "rospider.usd" if args_cli.fix_base else "rospider_float.usd"


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
        usd_file_name=_usd_name(),
        force_usd_conversion=True,
        # 센서 융합 과제에서는 보행이 목표가 아니다. base 를 고정해 두면 다리 제어 없이도
        # 로봇이 서 있고, 자세(기울기)를 우리가 원하는 값으로 정확히 줄 수 있다.
        # 보행까지 가려면 False 로 바꾸고 다리 PD 게인부터 다시 잡아야 한다.
        fix_base=args_cli.fix_base,
        # URDF 의 루트는 base_footprint(더미) 다. base_link 를 루트로 지정하면 merge 후에도
        # prim 이름이 base_link 로 남아 센서 경로를 적기 편하다. 다만 이렇게 트리를
        # 다시 뿌리내리면 임포터가 만드는 root_joint 가 꼬일 수 있다
        # ("Failed to create articulation at: .../root_joint").
        # 그때는 --root-link '' 로 두고, 바뀐 바디 이름을 --base_body 로 넘기면 된다.
        root_link_name=args_cli.root_link or None,
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
    """USD 구조를 찍는다. 물리를 안 돌리므로 아티큘레이션이 깨져 있어도 안전하다.

    Isaac Lab 의 `Articulation` 은 `UsdPhysics.ArticulationRootAPI` 가 붙은 prim 을
    **정확히 하나** 찾아서 PhysX 뷰를 만든다. 그게 어디에 붙어 있는지가
    "Failed to create articulation at: ..." 을 푸는 첫 단서다.
    """
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path))
    bodies, joints, fixed, roots = [], [], [], []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
        if prim.IsA(UsdPhysics.Joint):
            (fixed if prim.IsA(UsdPhysics.FixedJoint) else joints).append(prim.GetName())

    default_prim = stage.GetDefaultPrim()
    print(f"\n[기본 prim] {default_prim.GetPath() if default_prim else '(없음)'}")
    print(f"\n[아티큘레이션 루트 {len(roots)}개] {roots or '(없음!)'}")
    if len(roots) != 1:
        print(
            "  ** Isaac Lab 은 ArticulationRootAPI 가 붙은 prim 이 정확히 하나여야 한다. **\n"
            "     0개면 변환이 잘못된 것이고, 2개 이상이면 어느 쪽을 쓸지 몰라 실패한다."
        )
    print(f"\n[리짓바디 {len(bodies)}개] {', '.join(sorted(bodies))}")
    print(f"\n[움직이는 조인트 {len(joints)}개] {', '.join(sorted(joints))}")
    print(f"\n[고정 조인트 {len(fixed)}개] {', '.join(sorted(fixed))}")
    print(
        "\n센서 prim 경로는 위 리짓바디 이름에서 고른다."
        " base_link 와 link4 가 보이면 rospider_cfg.py 기본값이 맞다."
        " 다르면 run_fusion_demo.py 에 --base_body / --arm_body 로 넘기면 된다."
    )


def main() -> None:
    usd_path = convert()
    report(usd_path)

    if not args_cli.view:
        print("\n창으로 보려면:  python tasks/sensor_fusion/02_urdf_to_usd.py --view")
        return

    print("\n[GUI] 씬에 세워 본다. 창을 닫으면 종료된다.")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/light", light_cfg)
    robot_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    robot_cfg.func("/World/Robot", robot_cfg, translation=(0.0, 0.0, 0.12))

    sim.reset()
    sim.set_camera_view(eye=[0.9, 0.9, 0.6], target=[0.0, 0.0, 0.15])
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    import traceback

    _code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        _code = 1
    # Kit 종료가 매달리는 일이 잦아 워치독을 걸어 끝낸다(kit_exit.py 참고).
    shutdown(simulation_app, _code)
