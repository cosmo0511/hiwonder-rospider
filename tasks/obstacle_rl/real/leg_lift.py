#!/usr/bin/env python3
"""실물 ROSpider: 다리 하나를 femur 로 들고, 공중에서 다른 관절을 움직여 사람이 방향을 본다.

  python3 leg_lift.py LM tibla 0.2      ⚠️ LM 을 femur +0.3 으로 들고 tibla +0.2 → 3초 유지 → 되돌리고 내려놓음
나머지 다리 5개가 받친다. 가운데 다리를 쓰는 게 안정적이다.
"""
import sys, time
import rclpy
sys.path.insert(0, "/tmp")
from floor_check import C, SERVO, to_rad, to_pulse   # noqa: E402

leg, typ, amt = sys.argv[1], sys.argv[2], float(sys.argv[3])
LIFT = 0.3
rclpy.init()
n = C()
fem, j = f"femur_{leg}", f"{typ}_{leg}"
pf, pj = n.read([SERVO[fem][0], SERVO[j][0]])
af, aj = to_rad(pf, SERVO[fem][1]), to_rad(pj, SERVO[j][1])
print(f"{fem} {af:+.3f}, {j} {aj:+.3f}")
n.send({SERVO[fem][0]: to_pulse(af + LIFT, SERVO[fem][1])}, 1.0); time.sleep(1.3)
print(f"다리 듦 (femur +{LIFT})")
n.send({SERVO[j][0]: to_pulse(aj + amt, SERVO[j][1])}, 1.0); time.sleep(1.2)
r = n.read([SERVO[j][0]])[0]
print(f"{j} +{amt} 움직임 → 지금 {to_rad(r, SERVO[j][1]):+.3f} (변화 {to_rad(r, SERVO[j][1]) - aj:+.3f}). 3초 유지")
HOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
time.sleep(HOLD)
n.send({SERVO[j][0]: pj}, 1.0); time.sleep(1.2)
n.send({SERVO[fem][0]: pf}, 1.0); time.sleep(1.3)
print("되돌리고 내려놓음")
