import gymnasium as gym
import numpy as np
import rich


class PressTheLightUpButton(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, render_mode=None, size=2, game_length=1, hard_mode=False
    ):
        self.size, self.game_length = size, game_length
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(self.size,), dtype=int
        )
        self.action_space = gym.spaces.Discrete(self.size)
        self.step_num = 0
        self.done = True
        self.hard_mode = hard_mode

    def _get_obs(self):
        obs = [0] * self.size
        if self.step_num < len(self.state):
            obs[self.state[self.step_num]] = 1
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, self.size, size=self.game_length)
        self.step_num = 0
        self.done = False
        return (self._get_obs(), None)

    def render(self):
        rich.print(f"state: {self.state}, step_num: {self.step_num}")

    def step(self, action):
        target = (
            ((action + self.step_num) % self.size) if self.hard_mode else action
        )
        reward = int(target == self.state[self.step_num])
        self.step_num += 1
        if not reward:
            self.done = True
        return (
            self._get_obs(),
            reward,
            self.done,
            self.step_num >= self.game_length,
            {},
        )
