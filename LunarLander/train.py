import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder


class EpisodeAnnotatorWrapper(gym.Wrapper):
    global_episode_count = 0

    def __init__(self, env):
        super().__init__(env)
        self.current_episode = 0

    def reset(self, **kwargs):
        EpisodeAnnotatorWrapper.global_episode_count += 1
        self.current_episode = EpisodeAnnotatorWrapper.global_episode_count
        return super().reset(**kwargs)

    def render(self, *args, **kwargs):
        frame = self.env.render(*args, **kwargs)
        if frame is not None:
            frame = np.asarray(frame)
            text = f"Episode: {self.current_episode}"
            frame = frame.copy()
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame


def main():
    print("Creating environment...")
    env = make_vec_env(
        "LunarLander-v3",
        n_envs=4,
        wrapper_class=EpisodeAnnotatorWrapper,
    )
    env = VecVideoRecorder(
        env,
        "videos",
        record_video_trigger=lambda step: (step // 16384) % 6 == 0,
        video_length=1000,
    )

    print("Initializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    print("Training model...")
    model.learn(total_timesteps=1000000)

    model_name = "ppo-LunarLander-v3"
    print(f"Saving model to {model_name}.zip...")
    model.save(model_name)
    print("Training complete!")


if __name__ == "__main__":
    main()
