#!/usr/bin/env python3
"""체크포인트의 첫 층 입력을 늘린다 (새 관측 칸은 가중치 0 -> 처음 행동은 원래 정책과 같음).
옵티마이저 상태는 버린다. Kit 없이 돈다.

    python tasks/obstacle_rl/expand_ckpt.py <in.pt> <out.pt> --actor_add 209 --critic_add 215
"""

import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--actor_add", type=int, required=True)
ap.add_argument("--critic_add", type=int, required=True)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
for key, add in (("actor.0.weight", a.actor_add), ("critic.0.weight", a.critic_add)):
    w = sd[key]
    sd[key] = torch.cat([w, torch.zeros(w.shape[0], add, dtype=w.dtype)], dim=1)
    print(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")
torch.save({"model_state_dict": sd, "iter": ck.get("iter", 0), "infos": ck.get("infos")}, a.dst)
print(f"저장: {a.dst}")
