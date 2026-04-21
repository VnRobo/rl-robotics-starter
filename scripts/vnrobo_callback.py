"""SB3 callback that sends training metrics to VnRobo Fleet Monitor."""

from stable_baselines3.common.callbacks import BaseCallback


class VnRoboCallback(BaseCallback):
    """Sends episode reward and timestep to VnRobo every N steps."""

    def __init__(self, agent, robot_id: str, send_every: int = 10_000):
        super().__init__()
        self._agent = agent
        self._robot_id = robot_id
        self._send_every = send_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self._send_every == 0:
            mean_reward = None
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = float(
                    sum(ep["r"] for ep in self.model.ep_info_buffer)
                    / len(self.model.ep_info_buffer)
                )
            self._agent.send_heartbeat(
                status="training",
                metadata={
                    "step": self.num_timesteps,
                    "mean_reward": mean_reward,
                    "robot": self._robot_id,
                },
            )
        return True
