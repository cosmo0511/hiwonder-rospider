#!/usr/bin/env python3
"""실물 ROSpider: 바닥에 선 채로 관절 방향을 센서로 확인한다 (사람이 볼 필요 없음).

  python3 floor_check.py sense                    1초 동안 IMU · 바닥 거리만 읽기 (안 움직임)
  python3 floor_check.py move femur all           ⚠️ femur 6개를 +0.1 rad(드라이버 각도) 1초에 걸쳐 움직이고 측정 후 복귀
  python3 floor_check.py move tibla left          왼쪽 3개만. all / left / right / front / rear / 다리 하나(LF ... RR)

측정: IMU 가속도 평균(기울기), 움직이는 동안 자이로 적분(회전), 뎁스 카메라 화면 가운데 바닥 거리(몸 높이).
URDF 기준 +0.1 rad 이면 (공중에서) coxa: 왼발 뒤 · 오른발 앞, femur: 발 위로, tibla: 발 바깥으로.
바닥에 선 채라면 femur 전부 + → 몸이 내려감(바닥 거리 줄어듦), coxa 전부 + → 발이 붙어 있으니 몸이 위에서 봐서 시계 방향(오른쪽)으로 돈다 (자이로 z -).
"""
import sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, Image
from ros_robot_controller_msgs.srv import GetBusServoState
from ros_robot_controller_msgs.msg import GetBusServoCmd, ServosPosition, ServoPosition

LEGS = ["LF", "LM", "LR", "RF", "RM", "RR"]
SIDES = {"all": LEGS, "left": ["LF", "LM", "LR"], "right": ["RF", "RM", "RR"],
         "front": ["LF", "RF"], "rear": ["LR", "RR"], **{l: [l] for l in LEGS}}
SERVO = {"coxa_LF": (5, True), "femur_LF": (3, True), "tibla_LF": (1, True),
         "coxa_LM": (11, True), "femur_LM": (9, True), "tibla_LM": (7, True),
         "coxa_LR": (17, True), "femur_LR": (15, True), "tibla_LR": (13, True),
         "coxa_RF": (6, True), "femur_RF": (4, False), "tibla_RF": (2, False),
         "coxa_RM": (12, False), "femur_RM": (10, False), "tibla_RM": (8, False),
         "coxa_RR": (18, True), "femur_RR": (16, False), "tibla_RR": (14, False)}
TICK = 240.0 / 1000.0 * np.pi / 180.0
STEP = 0.10
DEPTH_TOPIC = "/depth_cam/depth/image_raw"


def to_rad(p, f):
    return ((500 - p) if f else (p - 500)) * TICK


def to_pulse(rad, f):
    raw = rad / TICK
    return int(round(min(1000, max(0, 500 - raw if f else 500 + raw))))


class C(Node):
    def __init__(self):
        super().__init__("floor_check")
        self.cli = self.create_client(GetBusServoState, "/ros_robot_controller/bus_servo/get_state")
        self.pub = self.create_publisher(ServosPosition, "/ros_robot_controller/bus_servo/set_position", 1)
        self.imu, self.depth = [], []
        for rel in (ReliabilityPolicy.BEST_EFFORT, ReliabilityPolicy.RELIABLE):
            q = QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=10)
            self.create_subscription(Imu, "/ros_robot_controller/imu_raw", self.cb_imu, q)
            self.create_subscription(Image, DEPTH_TOPIC, self.cb_depth, q)
        if not self.cli.wait_for_service(timeout_sec=5.0):
            sys.exit("get_state 서비스 없음 (ros_robot_controller 노드가 꺼져 있다)")

    def cb_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.imu.append((time.monotonic(), a.x, a.y, a.z, w.x, w.y, w.z))

    def cb_depth(self, m):
        if m.encoding not in ("16UC1", "mono16"):
            return
        d = np.frombuffer(bytes(m.data), dtype=np.uint16).reshape(m.height, m.width)
        h, w = d.shape
        c = d[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5].astype(np.float32)
        c = c[c > 0]
        if c.size:
            self.depth.append((time.monotonic(), float(np.median(c))))   # mm

    def collect(self, secs):
        self.imu.clear(); self.depth.clear()
        t = time.monotonic()
        while time.monotonic() - t < secs:
            rclpy.spin_once(self, timeout_sec=0.02)
        a = np.array(self.imu) if self.imu else np.zeros((0, 7))
        # 두 QoS 로 같은 메시지가 두 번 올 수 있어 시간 간격 0.5 ms 미만은 하나로
        if len(a) > 1:
            a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0.0005])]
        d = np.array([x[1] for x in self.depth]) if self.depth else np.zeros(0)
        return a, d

    def read(self, ids):
        req = GetBusServoState.Request()
        for i in ids:
            c = GetBusServoCmd(); c.id = i; c.get_position = 1
            req.cmd.append(c)
        f = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=3.0)
        r = f.result()
        if r is None or not r.success or len(r.state) < len(ids):
            sys.exit("서보 읽기 실패")
        return [int(s.position[-1]) for s in r.state]

    def send(self, pulses, duration):
        t = time.time()
        while self.pub.get_subscription_count() == 0 and time.time() - t < 5.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.pub.get_subscription_count() == 0:
            sys.exit("서보 명령을 받는 노드가 없다")
        m = ServosPosition(); m.duration = float(duration)
        for i, p in pulses.items():
            s = ServoPosition(); s.id = i; s.position = p
            m.position.append(s)
        self.pub.publish(m)


def summary(tag, a, d):
    acc = a[:, 1:4].mean(0) if len(a) else np.full(3, np.nan)
    dep = float(np.median(d)) if len(d) else float("nan")
    print(f"  [{tag}] 가속도 평균 {np.round(acc, 3)} (IMU {len(a)}개) · 바닥 거리 {dep:.0f} mm (뎁스 {len(d)}장)")
    return acc, dep


if __name__ == "__main__":
    rclpy.init()
    n = C()
    if sys.argv[1:2] == ["sense"]:
        summary("지금", *n.collect(1.5))
    elif sys.argv[1:2] == ["move"]:
        typ, side = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "all")
        assert typ in ("coxa", "femur", "tibla") and side in SIDES
        names = [f"{typ}_{l}" for l in SIDES[side]]
        p0 = n.read([SERVO[j][0] for j in names])
        print(f"{typ} {side}: 지금 각도 " + ", ".join(f"{j} {to_rad(p, SERVO[j][1]):+.3f}" for j, p in zip(names, p0)))
        acc0, dep0 = summary("전", *n.collect(1.0))
        tgt = {SERVO[j][0]: to_pulse(to_rad(p, SERVO[j][1]) + STEP, SERVO[j][1]) for j, p in zip(names, p0)}
        n.send(tgt, 1.0)
        a_mv, _ = n.collect(1.3)                   # 움직이는 동안: 자이로 적분
        acc1, dep1 = summary("후", *n.collect(1.0))
        p1 = n.read([SERVO[j][0] for j in names])
        print("  관절 변화(rad): " + ", ".join(f"{j} {to_rad(b, SERVO[j][1]) - to_rad(a, SERVO[j][1]):+.3f}"
                                             for j, a, b in zip(names, p0, p1)))
        if len(a_mv) > 1:
            rot = (a_mv[1:, 4:7] * np.diff(a_mv[:, 0])[:, None]).sum(0)
            print(f"  움직이는 동안 회전 (자이로 적분, rad): x {rot[0]:+.3f} y {rot[1]:+.3f} z {rot[2]:+.3f}")
        print(f"  기울기 변화 (가속도 후-전): {np.round(acc1 - acc0, 3)} · 바닥 거리 변화 {dep1 - dep0:+.0f} mm")
        n.send({SERVO[j][0]: p for j, p in zip(names, p0)}, 1.0)
        time.sleep(1.3)
        print("  원래 자세로 복귀함")
    else:
        sys.exit(__doc__)
