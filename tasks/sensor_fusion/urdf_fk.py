#!/usr/bin/env python3
"""URDF 체인을 따라 링크 포즈를 계산한다 (numpy 만 필요, Kit 불필요).

용도는 두 가지다.

1. **센서 외부 파라미터(extrinsic) 뽑기.** RosPider 의 뎁스 카메라는 base_link 가
   아니라 팔 끝(link4) 에 달려 있다. 융합 코드가 필요한 것은 "IMU 프레임 ->
   카메라 광학 프레임" 회전 하나인데, 그건 이 스크립트가 출력하는 두 포즈
   (imu_link, depth_cam_frame) 에서 바로 나온다.
2. **팔 자세 고르기.** 팔 조인트를 0 으로 두면 카메라가 천장을 본다. 전방을 보는
   조인트 조합을 찾을 때 `--joints` 로 값을 바꿔가며 forward 축을 확인한다.

사용법:
    python tasks/sensor_fusion/urdf_fk.py \
        --urdf assets/rospider_description/rospider.urdf \
        --joints joint2=1.0,joint3=-1.2,joint4=-0.9
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# 과제에서 관심 있는 링크. base_link 기준 포즈를 찍는다.
LINKS_OF_INTEREST = ("imu_link", "depth_cam_link", "depth_cam_frame", "lidar_link", "link4")


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF 의 rpy(고정축 X->Y->Z, 즉 R = Rz @ Ry @ Rx) 를 3x3 회전행렬로."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """로드리게스 공식. revolute 조인트의 회전."""
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    k = axis / norm
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 -> 쿼터니언 (w, x, y, z). Isaac Lab 의 쿼터니언 순서다."""
    trace = np.trace(rot)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


def load_joints(urdf_path: Path) -> dict[str, dict]:
    """URDF 의 조인트를 child 링크 이름으로 색인해 돌려준다."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        axis_el = j.find("axis")
        axis = [float(v) for v in (axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1").split()]
        joints[j.find("child").get("link")] = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "xyz": np.array(xyz),
            "rpy": np.array(rpy),
            "axis": np.array(axis),
        }
    return joints


def link_pose(link: str, joints: dict, q: dict[str, float], base: str = "base_link"):
    """base 기준 link 의 (위치 3, 회전 3x3). 체인을 거슬러 올라가 곱한다."""
    pos = np.zeros(3)
    rot = np.eye(3)
    chain = []
    cur = link
    while cur != base:
        if cur not in joints:
            raise SystemExit(f"{cur} 로 가는 조인트가 없습니다. 링크 이름을 확인하세요.")
        chain.append(joints[cur])
        cur = joints[cur]["parent"]
    for j in reversed(chain):
        step_rot = rpy_to_matrix(*j["rpy"])
        if j["type"] in ("revolute", "continuous"):
            step_rot = step_rot @ axis_angle_to_matrix(j["axis"], q.get(j["name"], 0.0))
        pos = pos + rot @ j["xyz"]
        rot = rot @ step_rot
    return pos, rot


def parse_joint_arg(text: str) -> dict[str, float]:
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        out[name.strip()] = float(value)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="base_link 기준 링크 포즈 계산")
    p.add_argument("--urdf", type=Path, default=Path("assets/rospider_description/rospider.urdf"))
    p.add_argument("--joints", default="", help="예: joint2=1.0,joint3=-1.2")
    p.add_argument("--links", default=",".join(LINKS_OF_INTEREST))
    args = p.parse_args()

    joints = load_joints(args.urdf)
    q = parse_joint_arg(args.joints)
    if q:
        print(f"팔 조인트: {q}")
    print(f"{'link':18s}{'pos (m)':34s}{'quat (w,x,y,z)':42s}")
    poses = {}
    for link in args.links.split(","):
        pos, rot = link_pose(link.strip(), joints, q)
        poses[link.strip()] = (pos, rot)
        quat = matrix_to_quat(rot)
        print(f"{link.strip():18s}{np.array2string(pos, precision=5, suppress_small=True):34s}"
              f"{np.array2string(quat, precision=5, suppress_small=True):42s}")

    # 카메라 광학 프레임의 축이 base_link 에서 어느 쪽을 향하는지. ROS 광학 규약은
    # +Z 가 시선(forward), +X 가 오른쪽, +Y 가 아래다.
    if "depth_cam_frame" in poses:
        _, rot = poses["depth_cam_frame"]
        print("\ndepth_cam_frame 축이 base_link 에서 가리키는 방향:")
        for label, col in zip(("X(right)", "Y(down) ", "Z(view) "), range(3)):
            print(f"  {label} -> {np.array2string(rot[:, col], precision=4, suppress_small=True)}")

    # 융합에 필요한 상수: IMU 프레임 -> 카메라 광학 프레임 회전
    if "imu_link" in poses and "depth_cam_frame" in poses:
        _, r_b_imu = poses["imu_link"]
        _, r_b_cam = poses["depth_cam_frame"]
        r_cam_imu = r_b_cam.T @ r_b_imu
        print("\nR_cam<-imu (융합 코드에 넣을 상수):")
        print(np.array2string(r_cam_imu, precision=6, suppress_small=True))
        print(f"쿼터니언 (w,x,y,z) = {np.array2string(matrix_to_quat(r_cam_imu), precision=6, suppress_small=True)}")


if __name__ == "__main__":
    main()
