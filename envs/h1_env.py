"""Unitree H1 humanoid locomotion environment."""

import os
import numpy as np
from envs.base_env import BaseLocomotionEnv

H1_JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
    "torso_joint",
]

H1_DEFAULT_POSE = np.array([
    0.0, 0.0, 0.98,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.4, 0.8, -0.4,
    0.0, 0.0, -0.4, 0.8, -0.4,
    0.0,
])

MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
    "/main/unitree_h1/h1.xml"
)


class H1Env(BaseLocomotionEnv):
    """Unitree H1 humanoid — forward walking task."""

    def __init__(self, **kwargs):
        self._model_path = self._download_model()
        super().__init__(**kwargs)

    def _get_model_path(self) -> str:
        return self._model_path

    def _get_joint_names(self):
        return H1_JOINT_NAMES

    def _get_default_pose(self) -> np.ndarray:
        return H1_DEFAULT_POSE.copy()

    def _get_min_base_height(self) -> float:
        return 0.4

    def _compute_reward(self) -> float:
        import mujoco
        vx = self._data.qvel[0]
        vel_reward = 1.0 - abs(vx - self.command_vel[0])
        base_quat = self._data.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, base_quat)
        upright = rot.reshape(3, 3)[2, 2]
        energy = -0.001 * np.sum(self._data.ctrl ** 2)
        return vel_reward + 0.5 * upright + energy + 1.0

    @staticmethod
    def _download_model() -> str:
        cache_dir = os.path.expanduser("~/.cache/vnrobo_rl/h1")
        model_path = os.path.join(cache_dir, "h1.xml")
        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(cache_dir, exist_ok=True)
            print("Downloading H1 MJCF from MuJoCo Menagerie...")
            urllib.request.urlretrieve(MENAGERIE_URL, model_path)
            print(f"Saved to {model_path}")
        return model_path
