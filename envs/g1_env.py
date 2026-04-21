"""Unitree G1 humanoid locomotion environment."""

import os
import numpy as np
from envs.base_env import BaseLocomotionEnv

# fmt: off
G1_JOINT_NAMES = [
    # Left leg
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
]

G1_DEFAULT_POSE = np.array([
    0.0, 0.0, 0.75,       # base pos
    1.0, 0.0, 0.0, 0.0,   # base quat
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    # Right leg (mirror)
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    # Waist
    0.0,
])
# fmt: on

MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
    "/main/unitree_g1/g1.xml"
)


class G1Env(BaseLocomotionEnv):
    """Unitree G1 humanoid — forward walking task.

    Reward: forward velocity + upright posture + energy efficiency.

    Usage:
        env = G1Env(render_mode="human")
        obs, _ = env.reset()
    """

    def __init__(self, **kwargs):
        self._model_path = self._download_model()
        super().__init__(**kwargs)

    def _get_model_path(self) -> str:
        return self._model_path

    def _get_joint_names(self):
        return G1_JOINT_NAMES

    def _get_default_pose(self) -> np.ndarray:
        return G1_DEFAULT_POSE.copy()

    def _get_min_base_height(self) -> float:
        return 0.3

    def _compute_reward(self) -> float:
        import mujoco

        # Forward velocity tracking
        vx = self._data.qvel[0]
        vel_reward = 1.0 - abs(vx - self.command_vel[0])

        # Upright posture (penalize tilt)
        base_quat = self._data.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, base_quat)
        rot = rot.reshape(3, 3)
        upright_reward = rot[2, 2]  # z-axis alignment → 1 when upright

        # Energy penalty
        energy_penalty = -0.001 * np.sum(self._data.ctrl ** 2)

        return vel_reward + 0.5 * upright_reward + energy_penalty + 1.0

    @staticmethod
    def _download_model() -> str:
        cache_dir = os.path.expanduser("~/.cache/vnrobo_rl/g1")
        model_path = os.path.join(cache_dir, "g1.xml")
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(cache_dir, exist_ok=True)
            print("Downloading G1 MJCF from MuJoCo Menagerie...")
            urllib.request.urlretrieve(MENAGERIE_URL, model_path)
            print(f"Saved to {model_path}")
        return model_path
