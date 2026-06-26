import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class NetworkEnv(gym.Env):
    """
    Custom Gymnasium environment simulating a network threat response task.

    Observation : Box(10,) float32 — simulated network flow features
    Actions     : Discrete(5)
                    0 = Monitor
                    1 = Alert
                    2 = RateLimit
                    3 = BlockIP
                    4 = Quarantine

    Reward structure:
        +10  correct block (BlockIP/Quarantine) on a real threat
        -5   blocking a benign flow (false positive)
        -2   only monitoring a real threat (missed action)
        +1   correctly monitoring a benign flow
    Episode ends after 100 steps.
    """

    metadata = {"render_modes": []}

    # Actions
    MONITOR    = 0
    ALERT      = 1
    RATE_LIMIT = 2
    BLOCK_IP   = 3
    QUARANTINE = 4

    MAX_STEPS = 100

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self._step_count = 0
        self._is_threat   = False
        self.np_random, _ = gym.utils.seeding.np_random()

    # ------------------------------------------------------------------
    def _sample_obs(self):
        """Generate a feature vector; threat flows have higher values."""
        base = self.np_random.random(10).astype(np.float32)
        if self._is_threat:
            # bias upper features to higher values to give the agent a signal
            base[5:] = np.clip(base[5:] + 0.4, 0.0, 1.0)
        return base

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._is_threat  = bool(self.np_random.random() > 0.5)
        obs = self._sample_obs()
        return obs, {}

    def step(self, action: int):
        is_block = action in (self.BLOCK_IP, self.QUARANTINE)
        is_monitor_or_alert = action in (self.MONITOR, self.ALERT, self.RATE_LIMIT)

        if self._is_threat:
            if is_block:
                reward = +10.0   # correct block of real threat
            else:
                reward = -2.0    # missed — only monitoring a real threat
        else:
            if is_block:
                reward = -5.0    # false positive — blocking benign
            else:
                reward = +1.0    # correctly monitoring benign

        self._step_count += 1
        terminated = False
        truncated  = self._step_count >= self.MAX_STEPS

        # Sample next flow (new threat status each step)
        self._is_threat = bool(self.np_random.random() > 0.5)
        obs = self._sample_obs()

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass


# Register so it can be created with gym.make()
register(
    id="NetworkEnv-v0",
    entry_point="rl_agent.environment:NetworkEnv",
    max_episode_steps=NetworkEnv.MAX_STEPS,
)
