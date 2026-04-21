# rl-robotics-starter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)

RL training baselines for legged robots — **Go2**, **G1**, **H1** — using MuJoCo + Stable-Baselines3. No GPU required to get started.

```bash
pip install -r requirements.txt
python scripts/train.py --robot go2 --timesteps 1_000_000
```

Models download automatically from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on first run.

## Supported Robots

| Robot | Type | DOF | Task | Status |
|-------|------|-----|------|--------|
| [Unitree Go2](https://www.unitree.com/go2/) | Quadruped | 12 | Forward locomotion | ✅ |
| [Unitree G1](https://www.unitree.com/g1/) | Humanoid | 13+ | Walking | ✅ |
| [Unitree H1](https://www.unitree.com/h1/) | Humanoid | 11 | Walking | ✅ |
| VNR-WH1 (VnRobo) | Wheeled Humanoid | 18 | Loco-manipulation | 🔜 |

## Install

```bash
git clone https://github.com/VnRobo/rl-robotics-starter
cd rl-robotics-starter
pip install -r requirements.txt
```

## Train

```bash
# Unitree Go2 — PPO (default, no GPU needed)
python scripts/train.py --robot go2 --timesteps 1_000_000

# Unitree G1 — SAC, 8 parallel envs
python scripts/train.py --robot g1 --algo sac --n-envs 8 --timesteps 2_000_000

# With real-time monitoring on VnRobo dashboard
python scripts/train.py --robot go2 --monitor --vnrobo-key YOUR_KEY
```

## Visualize Trained Policy

```bash
python scripts/enjoy.py --robot go2 --model checkpoints/go2_ppo_final
```

## Monitor Training in Real-Time

Track your training robots on **[app.vnrobo.com](https://app.vnrobo.com)** — free for 3 robots:

```bash
export VNROBO_API_KEY=your-key
python scripts/train.py --robot g1 --monitor
```

The dashboard shows episode reward, training step, and robot status updated every 10k steps.

## Project Structure

```
rl-robotics-starter/
├── envs/
│   ├── base_env.py          # Gymnasium base class (MuJoCo + PD control)
│   ├── go2_env.py           # Unitree Go2 quadruped
│   ├── g1_env.py            # Unitree G1 humanoid
│   └── h1_env.py            # Unitree H1 humanoid
├── scripts/
│   ├── train.py             # Main training script (PPO / SAC)
│   ├── enjoy.py             # Load policy + render
│   └── vnrobo_callback.py   # SB3 callback → VnRobo dashboard
├── configs/                 # Hyperparameter configs (YAML)
└── requirements.txt
```

## Reward Design

All environments share a common reward structure:

| Component | Weight | Description |
|-----------|--------|-------------|
| Velocity tracking | 1.0 | Match target forward velocity |
| Upright posture | 0.5 | Penalize tilt (humanoids only) |
| Energy efficiency | -0.001 | Minimize torque² |
| Alive bonus | 1.0 | Per step survival reward |

## Extending

Add a new robot by subclassing `BaseLocomotionEnv`:

```python
from envs.base_env import BaseLocomotionEnv

class MyRobotEnv(BaseLocomotionEnv):
    def _get_model_path(self): return "path/to/robot.xml"
    def _get_joint_names(self): return ["joint1", "joint2", ...]
    def _get_default_pose(self): return np.array([...])
    def _compute_reward(self): return ...
```

## Isaac Lab (GPU Training)

For large-scale GPU training with Isaac Lab, see [`configs/isaac_lab/`](configs/isaac_lab/) — coming soon.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

**Monitor your training robots for free → [app.vnrobo.com](https://app.vnrobo.com)**
