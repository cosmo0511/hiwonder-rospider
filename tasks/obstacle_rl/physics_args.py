"""평가 스크립트 공용 물리 덮어쓰기 인자. Kit 전에 import 하므로 Isaac 모듈을 쓰지 않는다.
적용은 env_cfg.physics_from_args."""


def add_physics_args(parser) -> None:
    """평가 스크립트 공용: 관절 속도 튐 시험용 물리 덮어쓰기 인자."""
    parser.add_argument("--vel_iters", type=int, default=None, help="물리: solver_velocity_iteration_count (기본 0)")
    parser.add_argument("--pos_iters", type=int, default=None, help="물리: solver_position_iteration_count (기본 4)")
    parser.add_argument("--leg_armature", type=float, default=None, help="물리: 다리 관절 armature [kg m^2]")
    parser.add_argument("--leg_vel_limit", type=float, default=None, help="물리: 다리 velocity_limit_sim [rad/s] (기본 6)")
    parser.add_argument("--dt", type=float, default=None, help="물리 스텝 [s] (기본 0.005). 제어 주기 0.02 s 는 유지")
