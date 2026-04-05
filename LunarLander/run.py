import gymnasium as gym
from stable_baselines3 import PPO


def main():
    print("Loading model...")
    model = PPO.load("ppo-LunarLander-v2")

    print("Creating environment for visualization...")
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    print("Running simulation...")
    for _ in range(1000):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("Simulation complete!")


if __name__ == "__main__":
    main()
