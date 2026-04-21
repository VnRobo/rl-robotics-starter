"""Load a trained policy and render it.

Usage:
    python scripts/enjoy.py --robot go2 --model checkpoints/go2_ppo_final
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["go2", "g1", "h1"], default="go2")
    p.add_argument("--model", required=True, help="Path to saved SB3 model (.zip)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    args = p.parse_args()

    from stable_baselines3 import PPO, SAC
    from envs import Go2Env, G1Env, H1Env

    ENVS = {"go2": Go2Env, "g1": G1Env, "h1": H1Env}
    ALGOS = {"ppo": PPO, "sac": SAC}

    env = ENVS[args.robot](render_mode="human")
    model = ALGOS[args.algo].load(args.model, env=env)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
