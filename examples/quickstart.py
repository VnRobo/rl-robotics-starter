"""Quick sanity check — instantiate each env and run 3 random steps."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs import Go2Env, G1Env, H1Env

ENVS = [("Go2", Go2Env), ("G1", G1Env), ("H1", H1Env)]

for name, EnvCls in ENVS:
    print(f"\n--- {name} ---")
    env = EnvCls()
    obs, _ = env.reset(seed=0)
    print(f"  obs shape : {obs.shape}")
    print(f"  action dim: {env.action_space.shape[0]}")
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"  step reward={reward:.3f}  terminated={terminated}")
    env.close()

print("\nAll environments OK.")
