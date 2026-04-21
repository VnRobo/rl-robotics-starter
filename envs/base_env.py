"""Base locomotion environment shared by all robots."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class BaseLocomotionEnv(gym.Env):
    """Base class for legged robot locomotion environments.

    Observation space: [joint_pos, joint_vel, base_lin_vel, base_ang_vel, projected_gravity, commands]
    Action space:      [target_joint_positions] (PD controller)

    Subclasses must implement:
        - _get_model_path() -> str
        - _get_joint_names() -> list[str]
        - _get_default_pose() -> np.ndarray
        - _compute_reward() -> float
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        control_freq: int = 50,       # Hz
        episode_length: int = 1000,   # steps
        command_vel: Tuple[float, float, float] = (0.5, 0.0, 0.0),  # vx, vy, yaw
    ):
        super().__init__()
        self.render_mode = render_mode
        self.control_freq = control_freq
        self.episode_length = episode_length
        self.command_vel = np.array(command_vel)
        self._step_count = 0

        # Lazy MuJoCo import
        self._model = None
        self._data = None
        self._viewer = None

        n_joints = len(self._get_joint_names())

        # Observation: joint_pos(n) + joint_vel(n) + lin_vel(3) + ang_vel(3) + gravity(3) + command(3)
        obs_dim = n_joints * 2 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Actions: target joint positions (normalized -1 to 1, scaled to joint range)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32
        )

        self._kp = 20.0   # position gain
        self._kd = 0.5    # velocity gain

    # ------------------------------------------------------------------ #
    # Gym interface                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_mujoco()
        import mujoco

        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = self._get_default_pose()
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._ensure_mujoco()
        import mujoco

        # Scale action to joint range
        n = len(self._get_joint_names())
        target = self._get_default_pose()[-n:] + action * 0.5

        # PD control
        current_pos = self._data.qpos[-n:]
        current_vel = self._data.qvel[-n:]
        torque = self._kp * (target - current_pos) - self._kd * current_vel
        self._data.ctrl[:] = np.clip(torque, -self._model.actuator_forcerange[:, 0], self._model.actuator_forcerange[:, 1])

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.episode_length

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        self._ensure_mujoco()
        import mujoco.viewer

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            import mujoco
            renderer = mujoco.Renderer(self._model, height=480, width=640)
            renderer.update_scene(self._data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_mujoco(self):
        if self._model is None:
            import mujoco
            self._model = mujoco.MjModel.from_xml_path(self._get_model_path())
            self._data = mujoco.MjData(self._model)

    def _get_obs(self) -> np.ndarray:
        import mujoco
        n = len(self._get_joint_names())
        joint_pos = self._data.qpos[-n:].copy()
        joint_vel = self._data.qvel[-n:].copy()

        # Base velocity in world frame → body frame
        base_quat = self._data.qpos[3:7]
        lin_vel = self._data.qvel[0:3].copy()
        ang_vel = self._data.qvel[3:6].copy()

        # Gravity vector projected onto base frame
        gravity_world = np.array([0.0, 0.0, -1.0])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, base_quat)
        rot = rot.reshape(3, 3)
        projected_gravity = rot.T @ gravity_world

        return np.concatenate([
            joint_pos, joint_vel, lin_vel, ang_vel,
            projected_gravity, self.command_vel,
        ]).astype(np.float32)

    def _is_terminated(self) -> bool:
        # Terminate if robot falls (base z too low)
        base_z = self._data.qpos[2]
        return bool(base_z < self._get_min_base_height())

    # ------------------------------------------------------------------ #
    # Subclass interface                                                    #
    # ------------------------------------------------------------------ #

    def _get_model_path(self) -> str:
        raise NotImplementedError

    def _get_joint_names(self):
        raise NotImplementedError

    def _get_default_pose(self) -> np.ndarray:
        raise NotImplementedError

    def _compute_reward(self) -> float:
        raise NotImplementedError

    def _get_min_base_height(self) -> float:
        return 0.1
