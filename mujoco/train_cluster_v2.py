import os
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.tune.registry import register_env
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import psutil
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Inicialización de Ray (automática para clúster o local)
if ray.is_initialized() == False:
    ray.init(address="auto")  # Ejecuta Ray en un cluster
    print(ray.available_resources())

#ray.init(num_cpus=12, include_dashboard=False)  # Ajusta `num_cpus` según tus necesidades
print(ray.available_resources())

# Configuración de parámetros
env_name = "Humanoid-v4"
model_name = "ppo"
hyperparams = "ts_12k_lr_0.0003"
log_dir = f"logs/{model_name}_{hyperparams}/"
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

    "num_workers": 8,
#    "num_envs_per_worker": 1,  # Un entorno por worker
#    "resources_per_worker": {
#        "CPU": 1},
    
    "log_level": "INFO",  # Agregar logs para depuración
    "framework": "torch",
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "lr": 0.0003,

    "logger_config": {"loggers": [ 
        JsonLoggerCallback, 
        CSVLoggerCallback, 
        TBXLoggerCallback]},
    "api_stack": {
        "enable_rl_module_and_learner": False, 
        "enable_env_runner_and_connector_v2": False},
}

algo = PPO(config=config)

# Entrenamiento
timesteps = 2_000_000
total_steps = 0

while total_steps < timesteps:
    results = algo.train()
    total_steps = results.get("timesteps_total", 0)

    # ==========================================================
    # MÉTRICAS CLAVE PARA EL AGENTE
    # ==========================================================

    # Recompensa promedio por episodio
    reward_mean = results['env_runners'].get('episode_reward_mean', None)
    if reward_mean is not None:
        writer.add_scalar('agent/reward_mean', reward_mean, total_steps)

    # Recompensa máxima y mínima por episodio
    reward_max = results['env_runners'].get('episode_return_max', None)
    reward_min = results['env_runners'].get('episode_return_min', None)
    if reward_max is not None:
        writer.add_scalar('agent/reward_max', reward_max, total_steps)
    if reward_min is not None:
        writer.add_scalar('agent/reward_min', reward_min, total_steps)

    # Longitud promedio de los episodios
    episode_len_mean = results['env_runners'].get('episode_len_mean', None)
    if episode_len_mean is not None:
        writer.add_scalar('agent/episode_len_mean', episode_len_mean, total_steps)

    # ==========================================================
    # MÉTRICAS DEL SISTEMA Y DISTRIBUCIÓN
    # ==========================================================

    # Uso promedio de CPU y RAM por worker
    cpu_ram_utilizations = []
    for worker_id in range(config['num_workers']):
        cpu_utilization = results.get('perf', {}).get(f'worker_{worker_id}_cpu_util_percent', None)
        ram_utilization = results.get('perf', {}).get(f'worker_{worker_id}_ram_util_percent', None)
        if cpu_utilization is not None:
            writer.add_scalar(f'system/worker_{worker_id}_cpu_util_percent', cpu_utilization, total_steps)
            cpu_ram_utilizations.append(f"Worker {worker_id} CPU Utilization: {cpu_utilization}%")
        if ram_utilization is not None:
            writer.add_scalar(f'system/worker_{worker_id}_ram_util_percent', ram_utilization, total_steps)
            cpu_ram_utilizations.append(f"Worker {worker_id} RAM Utilization: {ram_utilization}%")

    # Uso total promedio de CPU y RAM
    total_cpu_utilization = results.get('perf', {}).get('total_cpu_util_percent', None)
    total_ram_utilization = results.get('perf', {}).get('total_ram_util_percent', None)
    if total_cpu_utilization is not None:
        writer.add_scalar('system/total_cpu_util_percent', total_cpu_utilization, total_steps)
    if total_ram_utilization is not None:
        writer.add_scalar('system/total_ram_util_percent', total_ram_utilization, total_steps)

    # ==========================================================
    # MÉTRICAS GENERALES DEL SISTEMA
    # ==========================================================
    
    # Tiempo total de entrenamiento
    time_total_s = results.get("time_total_s", None)
    if time_total_s is not None:
        writer.add_scalar('system/time_total_s', time_total_s, total_steps)

    # Velocidad de aprendizaje (tiempo por iteración)
    time_this_iter_s = results.get("time_this_iter_s", None)
    if time_this_iter_s is not None:
        writer.add_scalar('system/time_this_iter_s', time_this_iter_s, total_steps)

    # Número total de pasos realizados
    writer.add_scalar('system/total_steps', total_steps, total_steps)

    # ==========================================================
    # DEPURACIÓN
    # ==========================================================

#    print(f"Resultados: {results}")

    print(f"Timesteps: {total_steps}, \n"
          f"Reward Mean: {reward_mean}, \n"
          f"Reward Max: {reward_max}, \n"
          f"Reward Min: {reward_min}, \n"
          f"Episode Len Mean: {episode_len_mean}, \n"
          f"Time Total: {time_total_s}, \n"
          f"Time This Iter: {time_this_iter_s}, \n"
          f"Total CPU Utilization: {total_cpu_utilization}, \n"
          f"Total RAM Utilization: {total_ram_utilization}, \n"
          + "\n".join(cpu_ram_utilizations))

# Cerrar el escritor de TensorBoard
writer.close()

# Guardar el modelo
checkpoint_dir = os.path.join(log_dir, f"{model_name}_checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)
algo.save_checkpoint(checkpoint_dir)

print(f"Model and logs saved at: {log_dir}")

algo.stop()