#!/usr/bin/env bash
# 학습이 끝난 실험의 평가만: 걸음새 평가(16마리) + 영상(4마리). 레포 루트에서 실행.
#   bash tasks/obstacle_rl/post_eval.sh 2f            # 그 실험의 가장 최근 로그, 마지막 체크포인트
#   bash tasks/obstacle_rl/post_eval.sh 2f 0.18 model_1500.pt
set -u
EXP="$1"; VX="${2:-0.18}"; MODEL="${3:-}"
export OMNI_KIT_ACCEPT_EULA=YES
OUT=outputs/obstacle_rl
RUN=$(ls -d logs/rsl_rl/rospider_obstacle/*_exp"$EXP" | sort | tail -1)
if [ -n "$MODEL" ]; then CK="$RUN/$MODEL"; else CK=$(ls "$RUN"/model_*.pt | sort -V | tail -1); fi
TAG=$(basename "$CK" .pt)
echo "[post_eval] 체크포인트 $CK"
python tasks/obstacle_rl/gait_eval.py --headless --num_envs 16 --vx "$VX" --controller policy \
  --checkpoint "$CK" --exp "$EXP" --name "exp${EXP}_${TAG}" > "$OUT/eval_exp${EXP}_${TAG}.log" 2>&1
echo "[post_eval] 평가 종료 코드 $?"
python tasks/obstacle_rl/gait_eval.py --headless --enable_cameras --video --num_envs 4 --vx "$VX" --controller policy \
  --checkpoint "$CK" --exp "$EXP" --name "exp${EXP}_${TAG}_video" > "$OUT/video_exp${EXP}_${TAG}.log" 2>&1
echo "[post_eval] 영상 종료 코드 $?"
cat "$OUT/gait_eval/exp${EXP}_${TAG}/metrics.txt"
