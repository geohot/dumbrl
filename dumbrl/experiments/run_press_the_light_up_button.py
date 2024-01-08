import rich
from stable_baselines3 import A2C

from dumbrl.environments import PressTheLightUpButton

if __name__ == "__main__":
    env = PressTheLightUpButton(size=16, game_length=10, hard_mode=False)
    # env = gym.make('LunarLander-v2', render_mode="rgb_array")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)

    vec_env = model.get_env()

    for i in range(10):
        rich.print(f"*** playing game {i}")
        done = False
        obs = vec_env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            rich.print(obs, action)
            obs, reward, done, info = vec_env.step(action)
            rich.print(reward, done)
            vec_env.render("human")
