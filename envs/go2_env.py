"""Unitree Go2 locomotion environment.

MJCF model auto-downloaded from MuJoCo Menagerie on first run.
"""

import os
import numpy as np
from envs.base_env import BaseLocomotionEnv

# fmt: off
GO2_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

GO2_DEFAULT_POSE = np.array([
    0.0, 0.0, 0.27,       # base pos
    1.0, 0.0, 0.0, 0.0,   # base quat
    # FR, FL, RR, RL — hip, thigh, calf
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
])
# fmt: on

MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
    "/main/unitree_go2/go2.xml"
)


class Go2Env(BaseLocomotionEnv):
    """Unitree Go2 quadruped — forward locomotion task.

    Reward: forward velocity tracking + energy penalty + alive bonus.

    Usage:
        env = Go2Env(render_mode="human")
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    def __init__(self, **kwargs):
        self._model_path = self._download_model()
        super().__init__(**kwargs)

    def _get_model_path(self) -> str:
        return self._model_path

    def _get_joint_names(self):
        return GO2_JOINT_NAMES

    def _get_default_pose(self) -> np.ndarray:
        return GO2_DEFAULT_POSE.copy()

    def _get_min_base_height(self) -> float:
        return 0.15

    def _compute_reward(self) -> float:
        # Forward velocity tracking
        vx = self._data.qvel[0]
        vel_reward = 1.0 - abs(vx - self.command_vel[0])

        # Energy penalty
        torque = self._data.ctrl.copy()
        energy_penalty = -0.001 * np.sum(torque ** 2)

        # Alive bonus
        alive = 1.0

        return vel_reward + energy_penalty + alive

    @staticmethod
    def _download_model() -> str:
        cache_dir = os.path.expanduser("~/.cache/vnrobo_rl/go2")
        model_path = os.path.join(cache_dir, "go2.xml")
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading Go2 MJCF from MuJoCo Menagerie...")
            urllib.request.urlretrieve(MENAGERIE_URL, model_path)
            print(f"Saved to {model_path}")
        return model_path
