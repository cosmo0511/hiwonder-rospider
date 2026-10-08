#!/usr/bin/env python3
"""rospider.xacro 를 단일 URDF 파일로 전개한다.

Isaac Sim / Kit 이 전혀 필요 없는 단계다. ROS 도 필요 없다. xacro 는 PyPI 판
(`pip install xacro`) 으로 충분하다. ROS 쪽 xacro 는 `$(find pkg)` 를 ament 로
해석하지만 우리는 ament 가 없으므로, 전개 전에 `$(find rospider_description)` 를
실제 디렉터리 경로로 직접 치환한다.

사용법:
    python tasks/sensor_fusion/01_xacro_to_urdf.py \
        --pkg assets/rospider_description \
        --out assets/rospider_description/rospider.urdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PKG_NAME = "rospider_description"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rospider.xacro -> rospider.urdf")
    p.add_argument(
        "--pkg",
        type=Path,
        default=Path("assets/rospider_description"),
        help="rospider_description 패키지 디렉터리 (urdf/, meshes/ 를 품고 있는 곳)",
    )
    p.add_argument(
        "--xacro",
        default="urdf/rospider.xacro",
        help="--pkg 기준 상대 경로. 기본값은 전체 로봇(다리+팔+센서 링크).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 URDF 경로. 기본값은 <pkg>/rospider.urdf",
    )
    p.add_argument(
        "--mesh-mode",
        choices=["relative", "absolute"],
        default="relative",
        help=(
            "메시 참조 방식. relative 는 'meshes/body/x.STL' 처럼 URDF 파일 기준"
            " 상대경로로 쓴다(권장, 컨테이너 경로가 바뀌어도 안 깨짐)."
            " absolute 는 'file:///...' 절대경로."
        ),
    )
    return p.parse_args()


def expand(xacro_path: Path, pkg_dir: Path) -> str:
    """xacro 를 전개해 URDF 문자열을 돌려준다."""
    try:
        import xacro  # noqa: PLC0415
    except ImportError:
        sys.exit("xacro 가 없습니다. `pip install xacro` 후 다시 실행하세요.")

    # $(find rospider_description) 를 실제 경로로 바꾼 사본을 만들어 전개한다.
    # xacro 는 include 를 파일 단위로 따라가므로, 패키지 전체를 한 번 훑어
    # 임시 디렉터리에 치환본을 만든다.
    import tempfile  # noqa: PLC0415

    abs_pkg = pkg_dir.resolve()
    find = r"\$\(find\s+" + re.escape(PKG_NAME) + r"\)"
    # include 는 치환본끼리 물려야 하므로 urdf/ 참조만 임시 디렉터리로 돌린다.
    # 그 외(메시 경로)는 원본 패키지 절대경로로 보낸다.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for src in (abs_pkg / "urdf").glob("*.xacro"):
            text = src.read_text(encoding="utf-8")
            text = re.sub(find + r"/urdf/", f"{tmp_dir}/", text)
            text = re.sub(find, str(abs_pkg), text)
            (tmp_dir / src.name).write_text(text, encoding="utf-8")
        doc = xacro.process_file(str(tmp_dir / xacro_path.name))
        return doc.toprettyxml(indent="  ")


def rewrite_mesh_paths(urdf: str, pkg_dir: Path, mode: str) -> str:
    """메시 filename 을 원하는 형식으로 통일한다."""
    abs_pkg = str(pkg_dir.resolve())
    # 전개 결과는 file:///abs/pkg/meshes/... 형태다.
    if mode == "relative":
        return urdf.replace(f"file://{abs_pkg}/", "").replace(f"{abs_pkg}/", "")
    return urdf.replace(f"file://{abs_pkg}/", f"file://{abs_pkg}/")


def sanity_check(urdf: str, pkg_dir: Path) -> None:
    """링크/조인트 수를 세고, 참조된 메시가 실제로 있는지 확인한다."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(urdf)
    links = [el.get("name") for el in root.findall("link")]
    joints = root.findall("joint")
    revolute = [j.get("name") for j in joints if j.get("type") == "revolute"]

    missing = []
    for mesh in root.iter("mesh"):
        fn = (mesh.get("filename") or "").replace("file://", "")
        path = Path(fn) if Path(fn).is_absolute() else pkg_dir / fn
        if not path.exists():
            missing.append(fn)

    print(f"[확인] link {len(links)}개, joint {len(joints)}개 (revolute {len(revolute)}개)")
    print(f"[확인] revolute 조인트: {', '.join(sorted(revolute))}")
    if missing:
        print(f"[경고] 참조는 있는데 파일이 없는 메시 {len(set(missing))}종:")
        for fn in sorted(set(missing)):
            print(f"        {fn}")
    else:
        print("[확인] 참조된 메시 파일 전부 존재")


def main() -> None:
    args = parse_args()
    pkg_dir = args.pkg
    if not (pkg_dir / "urdf" / "rospider.xacro").exists():
        sys.exit(
            f"{pkg_dir}/urdf/rospider.xacro 가 없습니다.\n"
            "먼저 bash tasks/sensor_fusion/00_fetch_description.sh 를 실행하세요."
        )

    out = args.out or (pkg_dir / "rospider.urdf")
    urdf = expand(pkg_dir / args.xacro, pkg_dir)
    urdf = rewrite_mesh_paths(urdf, pkg_dir, args.mesh_mode)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(urdf, encoding="utf-8")
    print(f"[완료] {out} ({len(urdf)} bytes)")
    sanity_check(urdf, pkg_dir)


if __name__ == "__main__":
    main()
