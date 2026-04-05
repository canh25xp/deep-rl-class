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
    # Run the simulation until the episode naturally ends (crash, safe landing, or timeout)
    while not done:
        # Ask the model to predict the optimal action based on the current observation
        # deterministic=True ensures it always picks what it thinks is the absolute best action
        action, _states = model.predict(observation, deterministic=True)

        # Execute the chosen action in the environment and receive the results of that action
        observation, reward, terminated, truncated, info = env.step(action)

        # 'terminated' means the game ended naturally (e.g. crash/landing)
        # 'truncated' means the game was artificially stopped (e.g. ran out of time steps limit)
        done = terminated or truncated

    # Close the environment specifically to release the allocated graphical resources
    env.close()
    print("Simulation complete!")


if __name__ == "__main__":
    main()
