# Huggy: Fetch the Stick

This project contains the standalone codebase to train Huggy, a Unity ML-Agents environment based on "Puppo The Corgi", using Proximal Policy Optimization (PPO).

## Prerequisites

- Linux (the environment executable included is for Linux)
- [uv](https://github.com/astral-sh/uv) (for ultra-fast Python environment management)

## Setup

A `setup.sh` script is provided to rapidly download the environment and set up the Python 3.10 virtual environment exactly required by `ml-agents`.

```bash
bash setup.sh
```

## Running the Project

### Train

To start training the agent, run the `train.py` script. The configuration for PPO is located in `config/ppo/Huggy.yaml`.

```bash
python train.py
```

_Note: You can pass `--resume` to continue training from your previous checkpoint._

### Evaluate

If you want to evaluate your agent's random baseline performance (rolling out average task reward calculations across episodes), you can run:

```bash
python evaluate.py
```

### Run (Visual Demonstration)

To run the Unity Environment with visual graphics rendered:

```bash
python run.py
```

_(Note: Natively testing learning checkpoints via `run.py` outside of the Unity Editor generally requires loading the `.onnx` models manually via `onnxruntime` or similar bindings)_
