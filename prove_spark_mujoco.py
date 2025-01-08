from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES
import time

import os
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.tune.registry import register_env
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter  # Importar TensorBoard

from pyspark.sql import SparkSession
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

if __name__ == "__main__":
    # Configuración de Spark
    spark = SparkSession \
        .builder \
        .appName("Ray on Spark Example") \
        .config("spark.task.cpus", "2") \
        .getOrCreate()

    # Configuración de Ray con Spark
    setup_ray_cluster(max_worker_nodes=2)

    # Inicializar Ray
    ray.init(ignore_reinit_error=True)


    # Configuración de parámetros
    env_name = "Humanoid-v4"
    model_name = "humanoid_ppo"
    hyperparams = "timesteps_200k_lr_0.0003"
    log_dir = f"RAY/logs/{model_name}_{hyperparams}/"
    os.makedirs(log_dir, exist_ok=True)

    # Registro del entorno
    def create_env(env_config):
        return gym.make(env_name)

    register_env(env_name, create_env)

    # Crear un escritor de TensorBoard
    writer = SummaryWriter(log_dir)

    # Configurar y entrenar el modelo
    config = {
        "env": env_name,
        "num_workers": 1,
        "framework": "torch",
        "logger_config": {
            "loggers": [
                JsonLoggerCallback,
                CSVLoggerCallback,
                TBXLoggerCallback  # Asegúrate de que TBXLoggerCallback está incluido
            ]
        },
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "lr": 0.0003,
        "api_stack": {"enable_rl_module_and_learner": False, "enable_env_runner_and_connector_v2": False},  # Deshabilitar la nueva API
    }

    algo = PPO(config=config)

    # Entrenamiento
    timesteps = 12_000
    total_steps = 0

    while total_steps < timesteps:
        results = algo.train()
        # Verificación de la presencia de claves y manejo de valores por defecto
        total_steps = results.get("timesteps_total", results.get("timesteps_this_iter", 0))
        print(f"Timesteps: {total_steps}, Reward: {results['env_runners'].get('episode_reward_mean', None)}")

        # Registro de métricas adicionales en TensorBoard
        reward_mean = results['env_runners'].get('episode_reward_mean', None)
        reward_max = results['env_runners'].get('episode_return_max', None)
        reward_min = results['env_runners'].get('episode_return_min', None)
        episode_len_mean = results['env_runners'].get('episode_len_mean', None)

        if reward_mean is not None and reward_mean != 'N/A':
            writer.add_scalar('training/reward_mean', reward_mean, total_steps)

        if reward_max is not None and reward_max != 'N/A':
            writer.add_scalar('training/reward_max', reward_max, total_steps)

        if reward_min is not None and reward_min != 'N/A':
            writer.add_scalar('training/reward_min', reward_min, total_steps)

        if episode_len_mean is not None and episode_len_mean != 'N/A':
            writer.add_scalar('training/episode_len_mean', episode_len_mean, total_steps)

        # Información más relevante de depuración
        print(f"Training Iteration: {results.get('training_iteration', 'N/A')}")
        print(f"Episode Len Mean: {results['env_runners'].get('episode_len_mean', 'N/A')}")
        print(f"Episode Reward Max: {results['env_runners'].get('episode_return_max', 'N/A')}")
        print(f"Episode Reward Min: {results['env_runners'].get('episode_return_min', 'N/A')}")

    # Cerrar el escritor de TensorBoard
    writer.close()

    # Definir el directorio del checkpoint correctamente
    checkpoint_dir = os.path.join(log_dir, f"{model_name}_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Asegurar que el directorio existe

    # Guardar el checkpoint en el directorio (no un archivo)
    algo.save_checkpoint(checkpoint_dir)

    print(f"Model and logs saved at: {log_dir}")

    algo.stop()