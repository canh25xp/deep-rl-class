import gymnasium as gym

# First, we create our environment called LunarLander-v2
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3", render_mode="human")

print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)  # Get a random observation
print("Observation Space Sample", env.observation_space.sample())  # Get a random observation
# Horizontal pad coordinate (x)
# Vertical pad coordinate (y)
# Horizontal speed (x)
# Vertical speed (y)
# Angle
# Angular speed
# If the left leg contact point has touched the land (boolean)
# If the right leg contact point has touched the land (boolean)

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action
# Action 0: Do nothing,
# Action 1: Fire left orientation engine,
# Action 2: Fire the main engine,
# Action 3: Fire right orientation engine.

# Then we reset this environment
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Take a random action
    print("Action taken:", action)

    # Do this action in the environment and get
    # next_state, reward, terminated, truncated and info
    observation, reward, terminated, truncated, info = env.step(action)
    # For each step, the reward:
    #   - Is increased/decreased the closer/further the lander is to the landing pad.
    #   - Is increased/decreased the slower/faster the lander is moving.
    #   - Is decreased the more the lander is tilted (angle not horizontal).
    #   - Is increased by 10 points for each leg that is in contact with the ground.
    #   - Is decreased by 0.03 points each frame a side engine is firing.
    #   - Is decreased by 0.3 points each frame the main engine is firing.
    # The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
    # An episode is considered a solution if it scores at least 200 points.

    # If the game is terminated (in our case we land, crashed) or truncated (timeout)
    if terminated or truncated:
        # Reset the environment
        print("Environment is reset")
        observation, info = env.reset()

env.close()
