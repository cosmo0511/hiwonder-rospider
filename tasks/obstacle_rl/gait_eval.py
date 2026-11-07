#!/usr/bin/env python3
"""걸음새를 숫자로 잰다. RL 정책과 기존 tripod 걸음(rospider_wasd.py)을 **같은 환경, 같은 조건**에서 비교한다.

    # 학습한 정책 (학습 때 준 --flat / --gait / --blind / --wasd_gains 를 똑같이)
    python tasks/obstacle_rl/gait_eval.py --headless --controller policy \
        --checkpoint logs/rsl_rl/rospider_obstacle/<날짜>/model_1499.pt --flat --blind --wasd_gains

    # 기존 tripod 걸음 (베이스라인). 게인은 항상 rospider_wasd.py 값
    python tasks/obstacle_rl/gait_eval.py --headless --controller tripod

    # 영상도 같이: --video --enable_cameras

평가 조건: 평지, 마찰 0.8 고정, 실물 질량(몸통 +1.54 kg = 3.2 kg), 밀기 / 관측 잡음 없음, 정면 출발.
정책 명령은 --vx 직진 + 방향 유지. tripod 는 obstacle_sweep.py 처럼 W 키(vx=1)를 계속 누른 것과 같다.

결과: outputs/obstacle_rl/gait_eval/<이름>/
  metrics.txt   env 평균 (표준편차): 속도, 옆/방향 틀어짐, tripod 일치율, 듀티비, 걸음 수, 발 높이, 미끄러짐
  gait.png      0 번 로봇의 발 접촉 타임라인 (검정 = 닿음). 위 3줄 = LF RM LR, 아래 3줄 = RF LM RR
  walk.mp4      --video 일 때
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RosPider 걸음새 평가")
parser.add_argument("--controller", choices=["policy", "tripod"], required=True)
parser.add_argument("--checkpoint", default=None, help="--controller policy 일 때 model_XXXX.pt")
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--vx", type=float, default=0.1, help="정책 전진 명령 (m/s)")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--friction", type=float, default=0.8)
parser.add_argument("--flat", action="store_true")
parser.add_argument("--gait", action="store_true")
parser.add_argument("--blind", action="store_true")
parser.add_argument("--wasd_gains", action="store_true")
parser.add_argument("--video", action="store_true", help="0 번 로봇을 따라가며 mp4 저장 (--enable_cameras 필요)")
parser.add_argument("--name", default=None, help="결과 폴더 이름. 기본은 체크포인트 / tripod")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.controller == "policy" and not args.checkpoint:
    parser.error("--controller policy 는 --checkpoint 가 필요하다")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 여기부터 Kit 이 뜬 뒤 ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tasks"))
from obstacle_rl.agents.rsl_rl_ppo_cfg import RoSpiderObstaclePPORunnerCfg  # noqa: E402
from obstacle_rl.env_cfg import RoSpiderObstacleEnvCfg, make_blind, make_flat, use_tripod_gait, use_wasd_gains  # noqa: E402

LEGS = ["LF", "LM", "LR", "RF", "RM", "RR"]
TRIPOD_A = ("LF", "RM", "LR")
TRIPOD_B = ("RF", "LM", "RR")
# 발끝 = tibla_* 프레임 기준 end_*_joint 원점 (URDF). 발끝은 지름 18 mm 구.
FOOT_TIP = {
    "LF": (0.11519, -0.0097341, 0.001174),
    "LM": (0.11519, -0.0097478, -0.0020999),
    "LR": (0.11519, -0.0097478, -0.00039987),
    "RF": (0.11519, 0.009749, 0.0010888),
    "RM": (0.1107, 0.011377, 0.016938),
    "RR": (0.11519, 0.009749, 0.00093881),
}
FOOT_R = 0.009
CONTACT_N = 1.0   # 이 힘(N)보다 크면 닿음

# ---- 기존 tripod 걸음. rospider_wasd.py 의 leg_targets() 를 그대로 옮김 (값을 바꾸지 말 것)
GAIT_HZ, COXA_AMP, FEMUR_LIFT, TIBIA_LIFT, SMOOTH_60HZ = 1.2, 0.35, 0.6, 0.3, 0.25
SIDE_SIGN = {"L": -1.0, "R": 1.0}


class TripodBaseline:
    """rospider_wasd.py 와 같은 열린 루프 tripod 걸음. 관절 목표를 절대각(rad)으로 만들어 행동으로 바꾼다."""

    def __init__(self, env: ManagerBasedRLEnv):
        term = env.action_manager.get_term("joint_pos")
        self.names = list(term._joint_names)
        self.offset = term._offset if torch.is_tensor(term._offset) else torch.tensor(term._offset, device=env.device)
        self.scale = term._scale if torch.is_tensor(term._scale) else torch.tensor(term._scale, device=env.device)
        self.dt = env.step_dt
        # wasd 는 60 Hz 에서 프레임마다 SMOOTH 만큼 따라간다 -> 이 env 의 제어 주기로 환산
        self.alpha = 1.0 - (1.0 - SMOOTH_60HZ) ** (60.0 * self.dt)
        self.phase = torch.zeros(env.num_envs, device=env.device)
        self.q = torch.zeros(env.num_envs, len(self.names), device=env.device)   # wasd 처럼 0 에서 시작

    def act(self, vx: float = 1.0, wz: float = 0.0) -> torch.Tensor:
        if abs(vx) + abs(wz) > 0:
            self.phase = (self.phase + GAIT_HZ * self.dt) % 1.0
        act = min(1.0, abs(vx) + abs(wz))
        des = torch.zeros_like(self.q)
        for j, name in enumerate(self.names):
            kind, leg = name.split("_")[0], name.split("_")[1]
            p = (self.phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
            swing = p < 0.5
            u = torch.where(swing, p / 0.5, (p - 0.5) / 0.5)
            frac = torch.where(swing, -torch.cos(math.pi * u), torch.cos(math.pi * u))
            lift = torch.where(swing, torch.sin(math.pi * u), torch.zeros_like(u))
            amp = max(-1.0, min(1.0, (vx - wz) if leg[0] == "L" else (vx + wz)))
            if kind == "coxa":
                des[:, j] = SIDE_SIGN[leg[0]] * COXA_AMP * amp * frac
            elif kind == "femur":
                des[:, j] = FEMUR_LIFT * act * lift
            else:
                des[:, j] = TIBIA_LIFT * act * lift
        self.q += self.alpha * (des - self.q)
        return (self.q - self.offset) / self.scale


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """불리언 1차원 배열에서 True 구간 [시작, 끝) 목록."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> None:
    name = args.name or (
        "tripod_baseline" if args.controller == "tripod"
        else f"{Path(args.checkpoint).resolve().parent.name}_{Path(args.checkpoint).stem}")
    out = REPO / "outputs" / "obstacle_rl" / "gait_eval" / name
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = RoSpiderObstacleEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    make_flat(env_cfg)
    if args.gait:
        use_tripod_gait(env_cfg)
    if args.blind:
        make_blind(env_cfg)
    if args.wasd_gains or args.controller == "tripod":
        use_wasd_gains(env_cfg)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.vx, args.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)
    cmd.heading_command = True
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    ev = env_cfg.events
    ev.push_robot = None
    ev.base_external_force_torque = None
    ev.physics_material.params["static_friction_range"] = (args.friction, args.friction)
    ev.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    ev.add_base_mass.params["mass_distribution_params"] = (1.54, 1.54)   # 3.2 - 1.66
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.episode_length_s = args.seconds + 5.0
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-0.55, -0.75, 0.40)
    env_cfg.viewer.lookat = (0.05, 0.0, 0.05)
    env_cfg.viewer.resolution = (1280, 720)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    obs, _ = env.reset()

    if args.controller == "policy":
        from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

        agent_cfg = RoSpiderObstaclePPORunnerCfg()
        agent_cfg.device = env_cfg.sim.device
        if args.blind:
            agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        pobs = wrapped.get_observations()
        pobs = pobs[0] if isinstance(pobs, tuple) else pobs
    else:
        tripod = TripodBaseline(env)

    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    rb_ids = [robot.find_bodies(f"tibla_{leg}")[0][0] for leg in LEGS]
    cs_ids = [sensor.find_bodies(f"tibla_{leg}")[0][0] for leg in LEGS]
    tip_off = torch.tensor([FOOT_TIP[leg] for leg in LEGS], device=env.device).unsqueeze(0).expand(env.num_envs, -1, -1)

    steps = int(args.seconds / env.step_dt)
    n = env.num_envs
    contact = np.zeros((steps, n, 6), dtype=bool)
    tip_z = np.zeros((steps, n, 6), dtype=np.float32)
    tip_vxy = np.zeros((steps, n, 6), dtype=np.float32)
    tilt = np.zeros((steps, n), dtype=np.float32)
    start = robot.data.root_pos_w.clone()
    ground_z = env.scene.env_origins[:, 2].cpu().numpy()
    writer = None
    if args.video:
        import imageio  # noqa: PLC0415

        every = max(1, round(1 / (25 * env.step_dt)))
        writer = imageio.get_writer(out / "walk.mp4", fps=round(1 / (every * env.step_dt)), codec="libx264", quality=7)

    for t in range(steps):
        with torch.inference_mode():
            if args.controller == "policy":
                pobs, _, _, _ = wrapped.step(policy(pobs))
                pobs = pobs[0] if isinstance(pobs, tuple) else pobs
            else:
                env.step(tripod.act(vx=1.0, wz=0.0))
        f = sensor.data.net_forces_w[:, cs_ids].norm(dim=-1)
        q = robot.data.body_quat_w[:, rb_ids].reshape(-1, 4)
        off_w = quat_apply(q, tip_off.reshape(-1, 3)).reshape(n, 6, 3)
        tip = robot.data.body_pos_w[:, rb_ids] + off_w
        vel = robot.data.body_lin_vel_w[:, rb_ids] + torch.cross(robot.data.body_ang_vel_w[:, rb_ids], off_w, dim=-1)
        g = robot.data.projected_gravity_b
        contact[t] = (f > CONTACT_N).cpu().numpy()
        tip_z[t] = (tip[..., 2] - FOOT_R).cpu().numpy() - ground_z[:, None]
        tip_vxy[t] = vel[..., :2].norm(dim=-1).cpu().numpy()
        tilt[t] = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))).cpu().numpy()
        if writer is not None and t % every == 0:
            writer.append_data(np.asarray(env.render()))
    if writer is not None:
        writer.close()

    # ---- 지표 (env 마다 -> 평균, 표준편차)
    d = (robot.data.root_pos_w - start).cpu().numpy()
    qw = robot.data.root_quat_w
    yaw = torch.rad2deg(torch.atan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]),
                                    1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2))).cpu().numpy()
    a_idx = [LEGS.index(x) for x in TRIPOD_A]
    b_idx = [LEGS.index(x) for x in TRIPOD_B]
    a_down, b_down = contact[..., a_idx].all(-1), contact[..., b_idx].all(-1)
    a_up, b_up = (~contact[..., a_idx]).all(-1), (~contact[..., b_idx]).all(-1)
    tripod_rate = ((a_down & b_up) | (b_down & a_up)).mean(0)
    all_down = contact.all(-1).mean(0)
    duty = contact.mean(0).mean(-1)
    steps_hz, clear, slip, stance_z = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    min_swing = 3   # 60 ms 보다 짧게 뜬 건 접촉 떨림으로 보고 걸음으로 안 센다
    for e in range(n):
        touch, peaks = 0, []
        for k in range(6):
            sw = [(s, t_end) for s, t_end in runs(~contact[:, e, k]) if t_end - s >= min_swing]
            touch += sum(1 for s, t_end in sw if s > 0 and t_end < steps)
            peaks += [tip_z[s:t_end, e, k].max() for s, t_end in sw]
        steps_hz[e] = touch / 6 / args.seconds
        clear[e] = np.mean(peaks) if peaks else 0.0
        m = contact[:, e]
        slip[e] = tip_vxy[:, e][m].mean() if m.any() else 0.0
        stance_z[e] = tip_z[:, e][m].mean() if m.any() else 0.0

    rows = [
        ("전진 속도 (m/s)", d[:, 0] / args.seconds, 3),
        ("옆으로 샌 거리 |y| (m)", np.abs(d[:, 1]), 3),
        ("방향 틀어짐 |yaw| (deg)", np.abs(yaw), 1),
        ("최대 기울기 (deg)", tilt.max(0), 1),
        ("tripod 일치율 (%)", tripod_rate * 100, 1),
        ("6발 모두 닿음 (%)", all_down * 100, 1),
        ("듀티비 (닿은 시간 비율)", duty, 2),
        ("다리당 걸음 수 (Hz)", steps_hz, 2),
        ("들린 발 최대 높이 평균 (cm)", clear * 100, 2),
        ("닿은 발 미끄러짐 (cm/s)", slip * 100, 2),
        ("[검산] 닿은 발끝 높이 (cm, 0 근처여야)", stance_z * 100, 2),
    ]
    head = [f"[{name}] controller={args.controller}  {args.seconds:.0f} s  env {n}개  마찰 {args.friction}  3.2 kg",
            f"  명령: {'vx ' + str(args.vx) + ' m/s 직진' if args.controller == 'policy' else 'wasd W 키 (vx=1)'}"]
    lines = head + [f"  {label:<24} {np.mean(v):+.{p}f}  (sd {np.std(v):.{p}f})" for label, v, p in rows]
    (out / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    # ---- 0 번 로봇 접촉 타임라인 (앞 5초)
    order = list(TRIPOD_A) + list(TRIPOD_B)
    show = min(steps, int(5.0 / env.step_dt))
    img = np.stack([contact[:show, 0, LEGS.index(leg)] for leg in order])
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.imshow(~img, aspect="auto", cmap="gray", interpolation="nearest",
              extent=[0, show * env.step_dt, 5.5, -0.5])
    ax.set_yticks(range(6), order)
    ax.axhline(2.5, color="tab:red", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{name}: foot contact (black = stance)  tripod {tripod_rate[0] * 100:.0f}%")
    fig.tight_layout()
    fig.savefig(out / "gait.png", dpi=120)
    print(f"[gait_eval] {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
