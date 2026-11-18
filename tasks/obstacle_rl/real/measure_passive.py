#!/usr/bin/env python3
"""실물 ROSpider 측정 (서보에 명령 안 보냄, 읽기만).
1) 다리 18 서보: 위치, 오프셋, 전압, 온도, 가동 범위, 토크 상태 한 번
2) 위치 읽기 시간: 18개 / 6개 / 1개 일괄, 각 N회
3) IMU (imu_raw) 주기, 정지 상태 평균 · 잡음
"""
import json, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from ros_robot_controller_msgs.srv import GetBusServoState
from ros_robot_controller_msgs.msg import GetBusServoCmd

JOINTS = ["coxa_LF", "coxa_LM", "coxa_LR", "coxa_RF", "coxa_RM", "coxa_RR",
          "femur_LF", "femur_LM", "femur_LR", "femur_RF", "femur_RM", "femur_RR",
          "tibla_LF", "tibla_LM", "tibla_LR", "tibla_RF", "tibla_RM", "tibla_RR"]
IDS = [5, 11, 17, 6, 12, 18, 3, 9, 15, 4, 10, 16, 1, 7, 13, 2, 8, 14]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
out = {}


class M(Node):
    def __init__(self):
        super().__init__("measure_passive")
        self.cli = self.create_client(GetBusServoState, "/ros_robot_controller/bus_servo/get_state")
        self.imu = []
        for rel in (ReliabilityPolicy.BEST_EFFORT, ReliabilityPolicy.RELIABLE):
            q = QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=10)
            self.create_subscription(Imu, "/ros_robot_controller/imu_raw", self.cb_imu, q)

    def cb_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.imu.append((time.monotonic(), a.x, a.y, a.z, w.x, w.y, w.z))

    def call(self, ids, **flags):
        req = GetBusServoState.Request()
        for i in ids:
            c = GetBusServoCmd(); c.id = i
            for k, v in flags.items():
                setattr(c, k, v)
            req.cmd.append(c)
        t = time.monotonic()
        f = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=3.0)
        return f.result(), time.monotonic() - t


rclpy.init()
n = M()
if not n.cli.wait_for_service(timeout_sec=5.0):
    print("get_state 서비스 없음"); sys.exit(1)

# 1) 한 번 전부
info = {}
for j, i in zip(JOINTS, IDS):
    r, _ = n.call([i], get_position=1, get_offset=1, get_temperature=1, get_position_limit=1)
    # get_voltage / get_torque_state 는 드라이버 버그(없는 함수 호출)로 노드가 죽어서 안 씀
    s = r.state[0] if r and r.success and r.state else None
    info[j] = None if s is None else dict(id=i, position=list(s.position), offset=list(s.offset),
                                          temperature=list(s.temperature),
                                          position_limit=list(s.position_limit))
out["servo_info"] = info
print(f"{'관절':10s} id  위치  오프셋  온도  범위")
for j, d in info.items():
    if d is None:
        print(f"{j:10s} 읽기 실패"); continue
    print(f"{j:10s} {d['id']:2d} {d['position']} {d['offset']} {d['temperature']} {d['position_limit']}")

# 2) 읽기 시간
timing = {}
for label, ids in (("18", IDS), ("6", IDS[:6]), ("1", IDS[:1])):
    ts, fail = [], 0
    for _ in range(N):
        r, dt = n.call(ids, get_position=1)
        if r is None or not r.success or len(r.state) < len(ids):
            fail += 1
        ts.append(dt)
    ts = np.array(ts) * 1000
    timing[label] = dict(mean_ms=float(ts.mean()), p50=float(np.median(ts)), p95=float(np.percentile(ts, 95)),
                         max=float(ts.max()), fail=fail, n=N)
    print(f"위치 읽기 {label:>2s}개 일괄: 평균 {ts.mean():.1f} ms, 중앙 {np.median(ts):.1f}, 95% {np.percentile(ts, 95):.1f}, "
          f"최대 {ts.max():.1f}, 실패 {fail}/{N}  -> {1000 / ts.mean():.1f} Hz")
out["read_timing"] = timing

# 3) IMU 5초
n.imu.clear()
t0 = time.monotonic()
while time.monotonic() - t0 < 5.0:
    rclpy.spin_once(n, timeout_sec=0.05)
a = np.array(n.imu)
if len(a) > 10:
    # 두 QoS 로 같은 메시지가 두 번 들어올 수 있어 시간 간격 0.5 ms 미만은 하나로
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0.0005])
    a = a[keep]
    dt = np.diff(a[:, 0])
    imu = dict(rate_hz=float(1 / dt.mean()), n=len(a), acc_mean=a[:, 1:4].mean(0).tolist(), acc_std=a[:, 1:4].std(0).tolist(),
               gyro_mean=a[:, 4:7].mean(0).tolist(), gyro_std=a[:, 4:7].std(0).tolist())
    print(f"IMU {imu['rate_hz']:.1f} Hz ({len(a)}개), 가속도 평균 {np.round(imu['acc_mean'], 3)} 잡음 {np.round(imu['acc_std'], 3)}")
    print(f"    자이로 평균 {np.round(imu['gyro_mean'], 4)} 잡음 {np.round(imu['gyro_std'], 4)}")
else:
    imu = None
    print(f"IMU 메시지 {len(a)}개뿐")
out["imu"] = imu
json.dump(out, open("/tmp/measure_passive.json", "w"), ensure_ascii=False, indent=1)
print("저장: /tmp/measure_passive.json")
