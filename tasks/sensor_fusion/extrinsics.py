"""센서 외부 파라미터를 **매 스텝** 계산한다. numpy 만 쓴다(Kit 불필요).

왜 매 스텝인가
--------------
RosPider 의 뎁스 카메라는 **팔 끝(link4)** 에 달려 있다. 팔이 움직이면 카메라 자세가
바뀌고, 그러면 융합에 쓰는 `R_cam<-imu`(IMU 프레임 -> 카메라 광학 프레임) 도 바뀐다.

처음에는 팔을 한 자세로 고정하고 이 값을 상수로 박아 뒀는데, 그러면 **팔을 움직이는
실험을 아예 못 한다.** 그래서 URDF 체인을 런타임에 풀어 매 스텝 다시 구한다.

이게 꼼수가 아닌 이유: 실기도 똑같이 한다. 팔 조인트 각도는 서보 엔코더로 알 수 있고
(`joint_states`), URDF 로 tf 를 계산해 카메라 자세를 얻는다. 여기서 읽는
`robot.data.joint_pos` 가 그 엔코더에 해당한다. **시뮬레이터의 정답 포즈를 몰래
가져다 쓰는 게 아니다** — 그걸 쓰면 IMU 가 필요 없어지므로 과제가 성립하지 않는다.

IMU 는 `base_link` 에 고정이라 `R_base<-imu` 는 진짜 상수다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urdf_fk import link_pose, load_joints

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5")


class ArmCameraExtrinsics:
    """URDF 체인에서 카메라/IMU 외부 파라미터를 뽑아 주는 계산기.

    Example:
        ext = ArmCameraExtrinsics("assets/rospider_description/rospider.urdf")
        r = ext.r_cam_imu({"joint2": 0.85, "joint3": -1.60, "joint4": -1.26})
    """

    def __init__(
        self,
        urdf_path: str | Path,
        base_link: str = "base_link",
        cam_link: str = "depth_cam_frame",
        imu_link: str = "imu_link",
    ) -> None:
        self.base_link = base_link
        self.cam_link = cam_link
        self._joints = load_joints(Path(urdf_path))
        # IMU 는 base_link 에 fixed 로 붙어 있다 -> 조인트 각도와 무관한 상수.
        _, self.r_base_imu = link_pose(imu_link, self._joints, {}, base=base_link)

    def cam_pose_in_base(self, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """주어진 팔 자세에서 base_link 기준 카메라 광학 프레임의 (위치, 회전행렬)."""
        return link_pose(self.cam_link, self._joints, q, base=self.base_link)

    def r_cam_imu(self, q: dict[str, float]) -> np.ndarray:
        """IMU 프레임 -> 카메라 광학 프레임 회전 (3x3). 융합에 바로 넣는 값."""
        _, r_base_cam = self.cam_pose_in_base(q)
        return r_base_cam.T @ self.r_base_imu

    def cam_height_in_base(self, q: dict[str, float]) -> float:
        """base_link 원점 기준 카메라 높이 [m]. floor_mode='assumed_height' 용."""
        pos, _ = self.cam_pose_in_base(q)
        return float(pos[2])

    @staticmethod
    def arm_joint_dict(joint_names: list[str], joint_pos: np.ndarray) -> dict[str, float]:
        """Isaac Lab 의 `robot.joint_names` / `robot.data.joint_pos[env]` 를 dict 로.

        팔 조인트만 추린다. 다리 18개는 카메라 체인에 없으므로 넣어도 무시되지만,
        불필요한 항목을 넘기지 않는 편이 디버깅할 때 읽기 쉽다.
        """
        out = {}
        for name in ARM_JOINT_NAMES:
            if name in joint_names:
                out[name] = float(joint_pos[joint_names.index(name)])
        return out
