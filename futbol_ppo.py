import os
import gymnasium as gym
import gymtonic
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment

env = gym.make('gymtonic/SoccerSingle-v0', max_speed=1.5, perimeter_side=8, goal_target='random', render_mode=None)

# Verify mesh file exists
#goal_path = "/home/rl/miniconda3/envs/rl/lib/python3.10/site-packages/gymtonic/envs/meshes/goal.obj"  # Update this path to the correct mesh file path
#if not os.path.exists(goal_path):
#    raise FileNotFoundError(f"Mesh file not found: {goal_path}")

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("dqn_soccer")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_soccer", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("rgb_array")





