# 최종 보행 정책 (땅 못 보는 CPG + RL)

실물 제어 주기에 맞춰 둘 중 하나를 쓴다. 결과와 구성은 [POLICY_SUMMARY.md](../POLICY_SUMMARY.md)

| 파일 | 제어 주기 | 환경 설정 | 원래 경로 | sha256 |
|---|---|---|---|---|
| `blind_3m_model_16096.pt` | 50 Hz (물리 0.005 s × 4) | `--exp 3m` | `logs/rsl_rl/rospider_obstacle/2026-11-10_05-05-18_exp3m/model_16096.pt` | `4553bb3a15c475f732909e232d91c1c4a8a33cd19c5a4186567ca10383fdd135` |
| `blind_3p_18hz_model_17096.pt` | 18.2 Hz (물리 0.005 s × 11) | `--exp 3p` | `logs/rsl_rl/rospider_obstacle/2026-11-11_05-15-16_exp3p/model_17096.pt` | `c58fcbccb003d96e1c47da5251b5deffb32817d68abf86f8f529c044580de332` |

3-p는 3-m과 입력 · 출력 크기가 같다. 다른 점은 제어 주기, 걸음 빠르기 범위(0.9 ~ 1.3배, 3-m은 0.5 ~ 1.3배), 학습 할인율뿐이다.
실물에서 3-p를 쓸 때는 CPG 박자 계산과 걸음 빠르기 변환도 0.9 ~ 1.3배로 맞춰야 한다.

평가 예:

```bash
python tasks/obstacle_rl/course_eval.py --headless --controller policy --exp 3m \
    --checkpoint tasks/obstacle_rl/policy/blind_3m_model_16096.pt --maps map1 map2 stairs
python tasks/obstacle_rl/course_eval.py --headless --controller policy --exp 3p \
    --checkpoint tasks/obstacle_rl/policy/blind_3p_18hz_model_17096.pt --maps map1 map2 stairs
python tasks/obstacle_rl/command_eval.py --headless --controller policy --exp 3p \
    --checkpoint tasks/obstacle_rl/policy/blind_3p_18hz_model_17096.pt
```

`--exp 3p`가 제어 주기를 18 Hz로 바꾼다. 다른 주기로 시험하려면 `--ctrl_hz 15`처럼 준다 (course_eval, gait_eval).

실물에 옮기는 건 아직 안 했다 (POLICY_SUMMARY의 "실물에 옮기기" 참고).
