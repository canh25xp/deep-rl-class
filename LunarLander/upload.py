#!/usr/bin/env python3

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub


def main():
    parser = argparse.ArgumentParser(description="Upload trained LunarLander model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Your Hugging Face repo ID in the format {username}/{repo_name} (e.g. YourUserName/ppo-LunarLander-v3)")
    args = parser.parse_args()

    env_id = "LunarLander-v3"
    model_name = "ppo-LunarLander-v3"
    model_architecture = "PPO"
    commit_message = "Upload PPO LunarLander-v3 trained agent"

    print(f"Loading model {model_name}...")
    model = PPO.load(model_name)

    print("Creating evaluation environment...")
    # The package_to_hub function actually evaluates the model in the background and records a video
    # which is why it requires an evaluation environment explicitly wrapped in a DummyVecEnv
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

    print(f"Pushing to Hugging Face Hub repository: {args.repo_id}...")
    package_to_hub(
        model=model,
        model_name=model_name,
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=eval_env,
        repo_id=args.repo_id,
        commit_message=commit_message,
    )
    print("Upload complete!")


if __name__ == "__main__":
    main()
