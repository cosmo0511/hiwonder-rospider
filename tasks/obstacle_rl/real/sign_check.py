#!/usr/bin/env python3
"""실물 ROSpider: 팔 자세 읽기 + 관절 종류별 방향 확인.

  python3 sign_check.py arm              팔·그리퍼 서보(19 ~ 24) 위치 읽기만 (안 움직임)
  python3 sign_check.py legs             다리 18개 위치 읽기만 (안 움직임)
  python3 sign_check.py move femur       ⚠️ femur 6개를 지금 위치에서 +0.15 rad(드라이버 각도) 1초에 걸쳐 움직이고
                                         2초 멈춘 뒤 1초에 걸쳐 원래 위치로 돌아온다. coxa / femur / tibla
로봇은 받침대에 올려 다리가 공중에 뜬 상태에서. 각도 변환은 드라이버(joint_position_controller.py)와 같은 식.
URDF 기준이라면 +0.15 rad 에서 coxa: 왼쪽 발 뒤로 · 오른쪽 발 앞으로, femur: 발 위로, tibla: 발 바깥으로.
"""
import sys, time
import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.srv import GetBusServoState
from ros_robot_controller_msgs.msg import GetBusServoCmd, ServosPosition, ServoPosition

LEGS = ["LF", "LM", "LR", "RF", "RM", "RR"]
# (id, flipped)  servo_controller.yaml: min > max 이면 flipped
SERVO = {"coxa_LF": (5, True), "femur_LF": (3, True), "tibla_LF": (1, True),
         "coxa_LM": (11, True), "femur_LM": (9, True), "tibla_LM": (7, True),
         "coxa_LR": (17, True), "femur_LR": (15, True), "tibla_LR": (13, True),
         "coxa_RF": (6, True), "femur_RF": (4, False), "tibla_RF": (2, False),
         "coxa_RM": (12, False), "femur_RM": (10, False), "tibla_RM": (8, False),
         "coxa_RR": (18, True), "femur_RR": (16, False), "tibla_RR": (14, False)}
TICK = 240.0 / 1000.0 * 3.141592653589793 / 180.0     # rad / 펄스
STEP = 0.15


def to_rad(p, flipped):
    return ((500 - p) if flipped else (p - 500)) * TICK


def to_pulse(rad, flipped):
    raw = rad / TICK
    return int(round(min(1000, max(0, 500 - raw if flipped else 500 + raw))))


class C(Node):
    def __init__(self):
        super().__init__("sign_check")
        self.cli = self.create_client(GetBusServoState, "/ros_robot_controller/bus_servo/get_state")
        self.pub = self.create_publisher(ServosPosition, "/ros_robot_controller/bus_servo/set_position", 1)
        if not self.cli.wait_for_service(timeout_sec=5.0):
            sys.exit("get_state 서비스 없음 (ros_robot_controller 노드가 꺼져 있다)")

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

    def send(self, pulses: dict, duration: float):
        # 노드를 만든 직후 publish 하면 연결 전이라 버려진다 (rl_walk.py 2026-11-10 기록)
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


def show(n, names):
    pos = n.read([SERVO[j][0] for j in names])
    for j, p in zip(names, pos):
        print(f"  {j:9s} id{SERVO[j][0]:2d}  펄스 {p:4d}  드라이버 각도 {to_rad(p, SERVO[j][1]):+.3f} rad")
    return pos


rclpy.init()
n = C()
mode = sys.argv[1] if len(sys.argv) > 1 else "legs"
if mode == "arm":
    ids = [19, 20, 21, 22, 23, 24]
    pos = n.read(ids)
    print("팔·그리퍼 펄스:", dict(zip(ids, pos)))
    print("부팅 자세 init_pose.yaml: {19: 500, 20: 650, 21: 40, 22: 215, 23: 500, 24: 500}")
elif mode == "legs":
    show(n, [f"{t}_{l}" for t in ("coxa", "femur", "tibla") for l in LEGS])
elif mode == "move":
    typ = sys.argv[2]
    assert typ in ("coxa", "femur", "tibla")
    names = [f"{typ}_{l}" for l in LEGS]
    print(f"[전] {typ}")
    p0 = show(n, names)
    tgt = {SERVO[j][0]: to_pulse(to_rad(p, SERVO[j][1]) + STEP, SERVO[j][1]) for j, p in zip(names, p0)}
    print(f"\n{typ} 6개 +{STEP} rad, 1초에 걸쳐 움직임")
    n.send(tgt, 1.0)
    time.sleep(1.5)
    print("[움직인 뒤]")
    p1 = show(n, names)
    for j, a, b in zip(names, p0, p1):
        print(f"  {j:9s} 변화 {to_rad(b, SERVO[j][1]) - to_rad(a, SERVO[j][1]):+.3f} rad")
    time.sleep(1.5)
    print("\n원래 위치로 1초에 걸쳐 복귀")
    n.send({SERVO[j][0]: p for j, p in zip(names, p0)}, 1.0)
    time.sleep(1.5)
    show(n, names)
else:
    sys.exit(__doc__)
