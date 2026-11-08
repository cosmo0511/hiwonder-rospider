#!/usr/bin/env bash
# 실험 하나를 끝까지: 학습 -> 걸음새 평가(16마리) -> 영상(4마리). 레포 루트에서 실행.
#   bash tasks/obstacle_rl/run_exp.sh 2c            # 1500 iteration
#   bash tasks/obstacle_rl/run_exp.sh 2c 800 0.12   # iteration 수, 평가 명령 속도(m/s)
#   bash tasks/obstacle_rl/run_exp.sh 3a 1500 0.18 logs/.../model_1999.pt   # 4번째 = 이어 학습할 체크포인트
# 결과: logs/rsl_rl/rospider_obstacle/<시각>_exp<이름>/, outputs/obstacle_rl/gait_eval/exp<이름>_*/
set -u
EXP="$1"; ITERS="${2:-1500}"; VX="${3:-0.18}"; RESUME="${4:-}"
RESUME_ARG=()
[ -n "$RESUME" ] && RESUME_ARG=(--resume "$RESUME")
export OMNI_KIT_ACCEPT_EULA=YES
OUT=outputs/obstacle_rl
mkdir -p "$OUT"

# 노트북이 절전으로 학습을 끊지 않게 잡아 둔다
systemd-inhibit --what=sleep:idle --who=rospider-rl --why="RL exp $EXP" \
  python tasks/obstacle_rl/train.py --headless --num_envs 2048 --max_iterations "$ITERS" --exp "$EXP" "${RESUME_ARG[@]}" ${TRAIN_EXTRA:-} \
  > "$OUT/train_exp$EXP.log" 2>&1
echo "[run_exp] 학습 종료 코드 $?"

RUN=$(ls -d logs/rsl_rl/rospider_obstacle/*_exp"$EXP" | sort | tail -1)
CK=$(ls "$RUN"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$CK" ]; then echo "[run_exp] 체크포인트가 없다 ($RUN). 학습 로그 확인: $OUT/train_exp$EXP.log"; exit 1; fi
echo "[run_exp] 체크포인트 $CK"
python tasks/obstacle_rl/gait_eval.py --headless --num_envs 16 --vx "$VX" --controller policy \
  --checkpoint "$CK" --exp "$EXP" --name "exp${EXP}_final" > "$OUT/eval_exp$EXP.log" 2>&1
echo "[run_exp] 평가 종료 코드 $?"
python tasks/obstacle_rl/gait_eval.py --headless --enable_cameras --video --num_envs 4 --vx "$VX" --controller policy \
  --checkpoint "$CK" --exp "$EXP" --name "exp${EXP}_final_video" > "$OUT/video_exp$EXP.log" 2>&1
echo "[run_exp] 영상 종료 코드 $?"
cat "$OUT/gait_eval/exp${EXP}_final/metrics.txt"
