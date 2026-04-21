"""Train a locomotion policy with Stable-Baselines3 + VnRobo monitoring.

Usage:
    python scripts/train.py --robot go2 --timesteps 1_000_000
    python scripts/train.py --robot g1 --timesteps 2_000_000 --monitor
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Train robot locomotion with SB3")
    p.add_argument("--robot", choices=["go2", "g1", "h1"], default="go2")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--monitor", action="store_true", help="Send metrics to VnRobo dashboard")
    p.add_argument("--vnrobo-key", default=os.environ.get("VNROBO_API_KEY", ""))
    return p.parse_args()


def make_env(robot: str, seed: int = 0):
    from envs import Go2Env, G1Env, H1Env
    ENVS = {"go2": Go2Env, "g1": G1Env, "h1": H1Env}
    env_cls = ENVS[robot]

    def _init():
        env = env_cls()
        env.reset(seed=seed)
        return env

    return _init


def main():
    args = parse_args()

    try:
        import stable_baselines3 as sb3
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("Install dependencies: pip install stable-baselines3[extra] mujoco")
        sys.exit(1)

    # VnRobo monitoring setup
    vnrobo_agent = None
    if args.monitor:
        if not args.vnrobo_key:
            print("Warning: --monitor requires VNROBO_API_KEY or --vnrobo-key")
        else:
            try:
                from vnrobo_agent import VnRoboAgent
                vnrobo_agent = VnRoboAgent(
                    api_key=args.vnrobo_key,
                    robot_id=f"{args.robot}-train",
                )
                print(f"VnRobo monitoring enabled for robot: {args.robot}-train")
            except ImportError:
                print("pip install vnrobo-agent to enable monitoring")

    # Create vectorized environments
    print(f"\nTraining {args.robot.upper()} with {args.algo.upper()}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Save dir: {args.save_dir}\n")

    vec_env = make_vec_env(make_env(args.robot, args.seed), n_envs=args.n_envs)
    eval_env = make_vec_env(make_env(args.robot, args.seed + 100), n_envs=1)

    os.makedirs(args.save_dir, exist_ok=True)

    # Algorithm config
    algo_cls = PPO if args.algo == "ppo" else SAC
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "verbose": 1,
        "seed": args.seed,
        "tensorboard_log": os.path.join(args.save_dir, "tb_logs"),
    }

    if args.algo == "ppo":
        model_kwargs.update({
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "learning_rate": 3e-4,
        })
    else:  # SAC
        model_kwargs.update({
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "learning_rate": 3e-4,
        })

    model = algo_cls(**model_kwargs)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // args.n_envs,
        save_path=args.save_dir,
        name_prefix=f"{args.robot}_{args.algo}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=os.path.join(args.save_dir, "eval"),
        eval_freq=50_000 // args.n_envs,
        n_eval_episodes=5,
        verbose=1,
    )

    # VnRobo callback
    callbacks = [checkpoint_cb, eval_cb]
    if vnrobo_agent:
        from scripts.vnrobo_callback import VnRoboCallback
        callbacks.append(VnRoboCallback(vnrobo_agent, args.robot))

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(args.save_dir, f"{args.robot}_{args.algo}_final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")

    if vnrobo_agent:
        vnrobo_agent.send_heartbeat(
            status="idle",
            metadata={"training": "complete", "robot": args.robot},
        )
        vnrobo_agent.stop()

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
