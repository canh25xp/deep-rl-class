import sys
from mlagents_envs.environment import UnityEnvironment


def main():
    # To run headlessly (no rendering)
    env_path = "./trained-envs-executables/linux/Huggy/Huggy"

    print(f"Starting UnityEnvironment headlessly from {env_path}...")
    env = UnityEnvironment(file_name=env_path, seed=1, side_channels=[], no_graphics=True)

    try:
        env.reset()
        # Find the behavior name
        behavior_name = list(env.behavior_specs.keys())[0]
        print(f"Behavior Name: {behavior_name}")
        spec = env.behavior_specs[behavior_name]

        total_episodes = 10
        total_rewards = []

        print("Evaluating agent using baseline (random actions)...")
        print("Note: To evaluate an ONNX model from python, an inference engine like onnxruntime is required.")

        for episode in range(total_episodes):
            env.reset()
            episode_reward = 0

            while True:
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # Assign actions for the agents that requested a decision
                if len(decision_steps) > 0:
                    actions = spec.action_spec.random_action(len(decision_steps))
                    env.set_actions(behavior_name, actions)

                    # Accumulate reward
                    for agent_id in decision_steps.agent_id:
                        episode_reward += decision_steps[agent_id].reward

                # Accumulate reward for terminal steps
                for agent_id in terminal_steps.agent_id:
                    episode_reward += terminal_steps[agent_id].reward

                env.step()

                if len(terminal_steps) > 0:
                    break

            print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
            total_rewards.append(episode_reward)

        avg_reward = sum(total_rewards) / total_episodes
        print(f"Average Reward over {total_episodes} episodes: {avg_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
