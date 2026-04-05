import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import demo_heuristic_lander

env = gym.make("LunarLander-v3", render_mode="human")

demo_heuristic_lander(env, render=True)
