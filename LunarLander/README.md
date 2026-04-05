# LunarLander-v2 Standalone Project

The `LunarLander-v2` Proximal Policy Optimization script from `unit1.ipynb` has been fully extracted into a standalone project at `/home/michael/projects/deep-rl-class/LunarLander`.

## Changes Made

1. **Created `requirements.txt`**: Added `stable-baselines3`, `swig`, `gymnasium[box2d]`, and `huggingface_sb3` using their latest versions, as requested.
2. **Created `train.py`**: Added the script to train a `MlpPolicy` PPO model for 1,000,000 timesteps across 16 vectorized environments, saving it as `ppo-LunarLander-v2.zip`.
3. **Created `evaluate.py`**: Added the script to evaluate the saved model over 10 deterministic episodes, computing mean reward and standard deviation.
4. **Created `run.py`**: Added the script with a `render_mode="human"` loop to let you watch the trained agent play the game.

## Instructions to Run

#### Step 1: Create and activate virtual environment

```bash
cd /home/michael/projects/deep-rl-class/LunarLander
python -m venv .venv
source .venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Train the Model

This will train the PPO model locally (approx. 20 minutes depending on your hardware).

```bash
python train.py
```

#### Step 4: Evaluate and Run

After training is finished, you can check its scores:

```bash
python evaluate.py
```

And finally, visually see it complete the landing:

```bash
python run.py
```
