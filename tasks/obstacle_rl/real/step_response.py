#!/usr/bin/env python3
"""실물 ROSpider: 관절 하나에 계단 명령을 주고 위치를 빠르게 읽어 서보 반응(지연, 속도, 도착 시간)을 잰다.
로봇을 들어 다리가 공중에 뜬 상태에서.

  python3 step_response.py tibla_LF 0.3 [duration]    ⚠️ +0.3 → 원래 → -0.3 → 원래, 각 1.5초. 한 서보만 5 ms 간격으로 읽는다
duration 은 서보 명령의 이동 시간 [s] (기본 0 = 최대 속도). 결과는 /tmp/step_<관절>_<duration>.csv
"""
import sys, time
import numpy as np
import rclpy
sys.path.insert(0, "/tmp")
from floor_check import C, SERVO, to_rad, to_pulse   # noqa: E402

j, amp = sys.argv[1], float(sys.argv[2])
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
rclpy.init()
n = C()
sid, fl = SERVO[j]
p0 = n.read([sid])[0]
a0 = to_rad(p0, fl)
print(f"{j} id{sid} 시작 {a0:+.3f} rad, 계단 ±{amp}, duration {dur}")
n.send({sid: p0}, 0.2)        # 구독 연결 확인 겸 제자리
time.sleep(0.3)
rows = []
t0 = time.monotonic()
for tgt in (a0 + amp, a0, a0 - amp, a0):
    ts = time.monotonic()
    n.send({sid: to_pulse(tgt, fl)}, dur)
    while time.monotonic() - ts < 1.5:
        t = time.monotonic()
        p = n.read([sid])[0]
        rows.append((t - t0, (t + time.monotonic()) / 2 - t0, tgt, to_rad(p, fl), ts - t0))
r = np.array(rows)
out = f"/tmp/step_{j}_{dur}.csv"
np.savetxt(out, r, delimiter=",", header="t_req,t_mid,target,pos,t_cmd", fmt="%.5f")
dt = np.diff(r[:, 1])
print(f"읽기 {len(r)}회, 간격 평균 {dt.mean() * 1000:.1f} ms")
# 각 계단: 움직이기 시작(1 펄스 이상 변화) 시간, 90 % 도달 시간, 최대 속도
for k, ts in enumerate(sorted(set(r[:, 4]))):
    s = r[r[:, 4] == ts]
    start, tgt = s[0, 3], s[0, 2]
    step = tgt - start
    if abs(step) < 0.05:
        continue
    t = s[:, 1] - ts
    moved = np.flatnonzero(np.abs(s[:, 3] - start) > 0.01)
    reach = np.flatnonzero(np.abs(s[:, 3] - start) >= 0.9 * abs(step))
    v = np.abs(np.diff(s[:, 3]) / np.diff(s[:, 1]))
    print(f"  계단 {start:+.3f} → {tgt:+.3f}: 움직이기 시작 {t[moved[0]] * 1000 if len(moved) else float('nan'):.0f} ms, "
          f"90 % 도달 {t[reach[0]] * 1000 if len(reach) else float('nan'):.0f} ms, 최대 속도 {v.max():.1f} rad/s, "
          f"마지막 오차 {s[-1, 3] - tgt:+.3f} rad")
print(f"저장: {out}")
