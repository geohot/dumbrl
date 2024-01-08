import random
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C

class PressTheLightUpButton(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, size=2, game_length=1, hard_mode=False):
        # Enhanced initialization with more complex game mechanics
        self.size, self.game_length = size, game_length
        self.observation_space = gym.spaces.Box(0, 1, shape=(self.size * 2,), dtype=int)
        self.action_space = gym.spaces.Discrete(self.size)
        self.hard_mode = hard_mode
        self.reset()

    def _get_obs(self):
        # Improved observation with history of actions and states
        obs = np.zeros(self.size * 2)
        obs[self.state[self.step_num]] = 1  # Current active button
        if self.step_num > 0:
            obs[self.size + self.previous_action] = 1  # Previous action taken
        return obs

    def reset(self, seed=None, options=None):
        # Reset the environment with a randomized state
        super().reset(seed=seed)
        self.state = np.random.choice(self.size, size=self.game_length, replace=True)
        self.step_num = 0
        self.done = False
        self.previous_action = None
        return self._get_obs()

    def step(self, action):
        # More complex step function with enhanced game mechanics
        target = ((action + self.step_num) % self.size) if self.hard_mode else action
        reward = 1 if target == self.state[self.step_num] else -1  # Negative reward for wrong actions
        self.previous_action = action
        self.step_num += 1
        self.done = self.step_num >= self.game_length
        return self._get_obs(), reward, self.done, {}

    def render(self):
        # Enhanced render for better visualization
        print(f"Game State: {self.state}, Current Step: {self.step_num}, Previous Action: {self.previous_action}")

def train_model(env, timesteps=50000):
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

def play_games(model, num_games=10):
    vec_env = model.get_env()
    for game in range(num_games):
        print(f"*** Playing Game {game + 1} ***")
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            vec_env.render()

if __name__ == "__main__":
    env = PressTheLightUpButton(size=16, game_length=10, hard_mode=True)
    model = train_model(env)
    play_games(model, num_games=10)

