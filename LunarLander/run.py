import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    print("Loading model...")
    model = PPO.load("ppo-LunarLander-v3")

    print("Creating environment for visualization...")
    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset()

    print("Running simulation...")

    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

    env.close()
    print("Simulation complete!")


if __name__ == "__main__":
    main()
