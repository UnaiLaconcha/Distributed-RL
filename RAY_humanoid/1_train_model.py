import os

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN

from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.tune.registry import register_env
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

    # Configuración de parámetros
    env_name = "Humanoid-v4"
    model_name = "dqn"
    hyperparams = "Ts200kLr0.0003"
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
        "num_workers": 1,
        "framework": "torch",
        "logger_config": {
            "loggers": [
                JsonLoggerCallback,
                CSVLoggerCallback,
                TBXLoggerCallback
            ]
        },
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "lr": 0.0003,
        "api_stack": {"enable_rl_module_and_learner": False, "enable_env_runner_and_connector_v2": False},
    }

    algo = PPO(config=config)

    # Entrenamiento
    timesteps = 200_000
    total_steps = 0

    while total_steps < timesteps:
        results = algo.train()
        total_steps = results.get("timesteps_total", results.get("timesteps_this_iter", 0))

        # ==========================================================
        # MÉTRICAS CLAVE PARA EL AGENTE
        # ==========================================================

        # 1. Recompensa promedio por episodio
        reward_mean = results['env_runners'].get('episode_reward_mean', None)
        if reward_mean is not None:
            writer.add_scalar('agent/reward_mean', reward_mean, total_steps)

        # 2. Recompensa máxima por episodio
        reward_max = results['env_runners'].get('episode_reward_max', None)
        if reward_max is not None:
            writer.add_scalar('agent/reward_max', reward_max, total_steps)

        # 3. Recompensa mínima por episodio
        reward_min = results['env_runners'].get('episode_reward_min', None)
        if reward_min is not None:
            writer.add_scalar('agent/reward_min', reward_min, total_steps)

        # 4. Longitud promedio de episodios
        episode_len_mean = results['env_runners'].get('episode_len_mean', None)
        if episode_len_mean is not None:
            writer.add_scalar('agent/episode_len_mean', episode_len_mean, total_steps)

        # 5. Número total de pasos realizados
        total_steps = results.get('timesteps_total', 0)
        writer.add_scalar('agent/total_steps', total_steps, total_steps)

        # 6. Divergencia de KL
        kl_divergence = results['info']['learner']['default_policy']['learner_stats'].get('kl', None)
        if kl_divergence is not None:
            writer.add_scalar('agent/kl_divergence', kl_divergence, total_steps)

        # ==========================================================
        # MÉTRICAS DE EFICIENCIA DE ENTRENAMIENTO
        # ==========================================================

        # 7. Pérdida total
        total_loss = results['info']['learner']['default_policy']['learner_stats'].get('total_loss', None)
        if total_loss is not None:
            writer.add_scalar('training/total_loss', total_loss, total_steps)

        # 8. Pérdida de política
        policy_loss = results['info']['learner']['default_policy']['learner_stats'].get('policy_loss', None)
        if policy_loss is not None:
            writer.add_scalar('training/policy_loss', policy_loss, total_steps)

        # 9. Gradiente global normalizado
        grad_norm = results['info']['learner']['default_policy']['learner_stats'].get('grad_gnorm', None)
        if grad_norm is not None:
            writer.add_scalar('training/grad_norm', grad_norm, total_steps)

        # 10. Velocidad de aprendizaje
        learning_throughput = results.get('num_env_steps_trained_throughput_per_sec', None)
        if learning_throughput is not None:
            writer.add_scalar('training/learning_throughput', learning_throughput, total_steps)

        # ==========================================================
        # MÉTRICAS DE EFICIENCIA DEL SISTEMA
        # ==========================================================

        # 11. Uso promedio de CPU
        cpu_utilization = results['perf'].get('cpu_util_percent', None)
        if cpu_utilization is not None:
            writer.add_scalar('system/cpu_utilization', cpu_utilization, total_steps)

        # 12. Uso promedio de RAM
        ram_utilization = results['perf'].get('ram_util_percent', None)
        if ram_utilization is not None:
            writer.add_scalar('system/ram_utilization', ram_utilization, total_steps)

        # 13. Velocidad de muestreo
        sampling_throughput = results.get('num_env_steps_sampled_throughput_per_sec', None)
        if sampling_throughput is not None:
            writer.add_scalar('system/sampling_throughput', sampling_throughput, total_steps)

        # 14. Latencia de muestreo
        sample_time_ms = results['timers'].get('sample_time_ms', None)
        if sample_time_ms is not None:
            writer.add_scalar('system/sample_time_ms', sample_time_ms, total_steps)

        # 15. Tiempo total de entrenamiento
        time_total_s = results.get('time_total_s', None)
        if time_total_s is not None:
            writer.add_scalar('system/time_total_s', time_total_s, total_steps)

        # ==========================================================
        # MÉTRICAS DE CALIDAD DEL MODELO
        # ==========================================================

        # 16. Valor explicado por la función de valor
        value_explained_var = results['info']['learner']['default_policy']['learner_stats'].get('vf_explained_var', None)
        if value_explained_var is not None:
            writer.add_scalar('model/value_explained_var', value_explained_var, total_steps)

        # ==========================================================
        # DEPURACIÓN
        # ==========================================================

        #print(f"Resultados: {results}")

        print(f"Timesteps: {total_steps}, \n"
            f"Reward Mean: {reward_mean}, \n"
            f"Reward Max: {reward_max}, \n"
            f"Reward Min: {reward_min}, \n"
            f"Episode Len Mean: {episode_len_mean}, \n"
            f"KL Divergence: {kl_divergence}, \n"
            f"Total Loss: {total_loss}, \n"
            f"Policy Loss: {policy_loss}, \n"
            f"Grad Norm: {grad_norm}, \n"
            f"Learning Throughput: {learning_throughput}, \n"
            f"CPU Utilization: {cpu_utilization}, \n"
            f"RAM Utilization: {ram_utilization}, \n"
            f"Sampling Throughput: {sampling_throughput}, \n"
            f"Sample Time (ms): {sample_time_ms}, \n"
            f"Total Training Time: {time_total_s}, \n"
            f"Value Explained Variance: {value_explained_var}\n")

    # Cerrar el escritor de TensorBoard
    writer.close()

    # Guardar el modelo
    checkpoint_dir = os.path.join(log_dir, f"{model_name}_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    algo.save_checkpoint(checkpoint_dir)

    print(f"Model and logs saved at: {log_dir}")

    algo.stop()
