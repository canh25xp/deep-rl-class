import sys
from mlagents_envs.environment import UnityEnvironment
import time


def main():
    env_path = "./trained-envs-executables/linux/Huggy/Huggy"

    print(f"Starting UnityEnvironment visually from {env_path}...")
    # Omitting 'no_graphics' enables visual rendering
    env = UnityEnvironment(file_name=env_path, seed=1, side_channels=[])

    try:
        env.reset()
        behavior_name = list(env.behavior_specs.keys())[0]
        spec = env.behavior_specs[behavior_name]

        print("Visualizing agent using a simple random policy.")
        print("Note: You must replace this manual action selection with your loaded ONNX model to visualize your trained agent.")

        episodes_to_run = 3

        for episode in range(episodes_to_run):
            env.reset()

            while True:
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                if len(decision_steps) > 0:
                    actions = spec.action_spec.random_action(len(decision_steps))
                    env.set_actions(behavior_name, actions)

                env.step()

                # Sleep briefly to slow down execution so it can be watched
                time.sleep(0.01)

                if len(terminal_steps) > 0:
                    print(f"Finished visual episode {episode + 1}.")
                    break

    finally:
        env.close()


if __name__ == "__main__":
    main()
