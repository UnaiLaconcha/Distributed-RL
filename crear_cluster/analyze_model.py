import os
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
import numpy as np
import torch

os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"

# Configuración de la ruta del modelo
checkpoint_path = "/home/rl/proyectos/proyRL/RAY/logs/humanoid_ppo_timesteps_200k_lr_0.0003/humanoid_ppo_checkpoint"

env_name = "Humanoid-v4"

# Registro del entorno
def create_env(env_config):
    return gym.make(env_name)

register_env(env_name, create_env)

# Verificar si el checkpoint existe
if not os.path.isdir(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

# Cargar el modelo
algo = PPO.from_checkpoint(checkpoint_path)

def process_obs(obs):
    if isinstance(obs, tuple):
        # Procesar cada elemento de la tupla por separado
        processed_obs = []
        for o in obs:
            if isinstance(o, np.ndarray):
                processed_obs.append(np.atleast_1d(o))  # Convertir a 1D si es necesario
            else:
                processed_obs.append(np.atleast_1d(np.array(o)))
        return np.concatenate(processed_obs)
    return np.atleast_1d(np.array(obs))  # Convertir a 1D si no es una tupla

# Evaluación de la política
def evaluate_agent(algo, env_name, n_episodes=10):
    env = gym.make(env_name)
    total_reward = 0
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            obs = process_obs(obs)  # Procesar la observación
            obs_tensor = torch.tensor(obs, dtype=torch.float32)  # Convertir a tensor
            action = algo.compute_single_action(obs_tensor)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        print(f"Episode {episode + 1}: Reward {episode_reward}")
        total_reward += episode_reward
    mean_reward = total_reward / n_episodes
    print(f"Mean Reward: {mean_reward}")
    return mean_reward

# Visualización del agente
def visualize_agent(algo, env_name, n_episodes=5):
    env = gym.make(env_name, render_mode="human")
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            obs = process_obs(obs)  # Procesar la observación
            obs_tensor = torch.tensor(obs, dtype=torch.float32)  # Convertir a tensor
            action = algo.compute_single_action(obs_tensor)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Reward {episode_reward}")

# Evaluar y visualizar
try:
    evaluate_agent(algo, env_name, n_episodes=10)
    n_visual_trials = int(input("Enter the number of episodes to visualize: "))
    visualize_agent(algo, env_name, n_episodes=n_visual_trials)
finally:
    algo.stop()
