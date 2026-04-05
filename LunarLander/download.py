#!/usr/bin/env python3

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub


def main():
    parser = argparse.ArgumentParser(description="Download and run a LunarLander model from Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="canh25xp/ppo-LunarLander-v3", help="Your Hugging Face repo ID (e.g. canh25xp/ppo-LunarLander-v3)")
    parser.add_argument("--filename", type=str, default="ppo-LunarLander-v3.zip", help="The name of the zip file in the repository")
    args = parser.parse_args()

    print(f"Downloading model '{args.filename}' from '{args.repo_id}'...")
    # This downloads the file from the Hub and returns the local path to it
    checkpoint = load_from_hub(repo_id=args.repo_id, filename=args.filename)

    print("Loading model...")
    print(checkpoint)
    # Load the PPO model from the downloaded checkpoint
    model = PPO.load(checkpoint)

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
