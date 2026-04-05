from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder


def main():
    print("Creating environment...")
    env = make_vec_env("LunarLander-v3", n_envs=4)
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
    )

    print("Training model...")
    model.learn(total_timesteps=1000000)

    model_name = "ppo-LunarLander-v3"
    print(f"Saving model to {model_name}.zip...")
    model.save(model_name)
    print("Training complete!")


if __name__ == "__main__":
    main()
