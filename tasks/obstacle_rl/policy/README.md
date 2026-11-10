# 최종 보행 정책 (3-m, 땅 못 보는 CPG + RL)

- `blind_3m_model_16096.pt`: rsl-rl 체크포인트 (원래 경로 `logs/rsl_rl/rospider_obstacle/2026-11-10_05-05-18_exp3m/model_16096.pt`)
- sha256 `4553bb3a15c475f732909e232d91c1c4a8a33cd19c5a4186567ca10383fdd135`
- 환경 설정은 `--exp 3m` (experiments.py). 결과와 구성은 [POLICY_SUMMARY.md](../POLICY_SUMMARY.md)

평가 예:

```bash
python tasks/obstacle_rl/course_eval.py --headless --controller policy --exp 3m \
    --checkpoint tasks/obstacle_rl/policy/blind_3m_model_16096.pt --maps map1 map2 stairs
python tasks/obstacle_rl/command_eval.py --headless --controller policy --exp 3m \
    --checkpoint tasks/obstacle_rl/policy/blind_3m_model_16096.pt
```

실물에 옮기는 건 아직 안 했다 (POLICY_SUMMARY의 "실물에 옮기기" 참고).
