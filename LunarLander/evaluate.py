import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    print("Loading model...")
    model = PPO.load("ppo-LunarLander-v3")

    print("Creating evaluation environment...")
    # Create the environment and wrap it with a Monitor to accurately log episode statistics (rewards/lengths)
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))

    print("Evaluating policy...")
    # Evaluate the policy over 10 deterministic episodes (no random exploration)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    # Target reward for "solving" LunarLander is typically a mean reward >= 200
    print(f"Evaluation complete: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
