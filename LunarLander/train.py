import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder


class EpisodeAnnotatorWrapper(gym.Wrapper):
    """
    A Gym environment wrapper that annotates rendered frames with the current episode number.
    It hooks into the `reset` and `render` methods to track episodes and overlay text.
    """

    global_episode_count = 0

    def __init__(self, env):
        super().__init__(env)
        self.current_episode = 0

    def reset(self, **kwargs):
        EpisodeAnnotatorWrapper.global_episode_count += 1
        self.current_episode = EpisodeAnnotatorWrapper.global_episode_count
        return super().reset(**kwargs)

    def render(self, *args, **kwargs):
        # Generate the original frame from the environment
        frame = self.env.render(*args, **kwargs)
        if frame is not None:
            frame = np.asarray(frame)
            text = f"Episode: {self.current_episode}"
            frame = frame.copy()
            # Overlay the episode text on the top-left of the image
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame


def main():
    print("Creating environment...")
    # Create a vectorized environment (16 parallel instances) for faster training
    env = make_vec_env(
        "LunarLander-v3",
        n_envs=16,
        wrapper_class=EpisodeAnnotatorWrapper,
    )
    env = VecVideoRecorder(
        env,
        "videos",
        record_video_trigger=lambda step: step % 6_250 == 0,
        video_length=500,
    )

    print("Initializing PPO model...")
    # Initialize the Proximal Policy Optimization (PPO) agent
    # We use an MlpPolicy (Multi-Layer Perceptron) because our input is a coordinate vector, not an image
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,  # Discount factor (how much the agent cares about long-term rewards)
        gae_lambda=0.98,  # Bias vs variance trade-off factor for Generalized Advantage Estimator
        ent_coef=0.01,  # Entropy coefficient (encourages exploration)
        verbose=1,
        device="cpu",  # Training forced on CPU (can be faster for lightweight vectorized envs)
    )

    print("Training model...")
    model.learn(total_timesteps=1_000_000)

    model_name = "ppo-LunarLander-v3"
    print(f"Saving model to {model_name}.zip...")
    model.save(model_name)
    print("Training complete!")


if __name__ == "__main__":
    main()
