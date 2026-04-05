import subprocess
import sys
import os


def main():
    venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin")
    mlagents_learn = os.path.join(venv_bin, "mlagents-learn")

    # Use python environment ml-agents-learn command
    if not os.path.exists(mlagents_learn):
        print(f"Executable {mlagents_learn} not found. Ensure you ran setup.sh and the .venv is activated.")
        mlagents_learn = "mlagents-learn"  # Fallback to path

    config_path = "./config/ppo/Huggy.yaml"
    env_path = "./trained-envs-executables/linux/Huggy/Huggy"
    run_id = "HuggyRun"

    args = [
        mlagents_learn,
        config_path,
        f"--env={env_path}",
        f"--run-id={run_id}",
        "--no-graphics",
    ]

    if "--resume" in sys.argv:
        args.append("--resume")

    if "--force" in sys.argv:
        args.append("--force")

    print(f"Running command: {' '.join(args)}")

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
