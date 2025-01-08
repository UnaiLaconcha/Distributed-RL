from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import gymnasium as gym
import os

# Registrar el entorno personalizado
def register_custom_env(env_config):
    env = gym.make("Humanoid-v4")
    return env

def main():
    # Desactivar la verificación estricta de métricas
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    # Registrar el entorno
    register_env("MyEnv", register_custom_env)

    # Configuración de PPO
    config = (
        PPOConfig()
        .environment("MyEnv")
        .framework("torch")  # Cambia a "tf" si prefieres TensorFlow
        .training(
            model={"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"},
            gamma=0.99,
            lr=0.001,
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=1,  # Evaluar cada iteración de entrenamiento
            evaluation_duration=10,  # Número de episodios por evaluación
        )
    ).to_dict()

    # Espacio de búsqueda
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.99),
        "model": {
            "fcnet_hiddens": tune.choice([[64, 64], [128, 128], [256, 256]]),
            "fcnet_activation": tune.choice(["relu", "tanh"]),
        },
    }

    # Ejecutar el entrenamiento con búsqueda de hiperparámetros
    analysis = tune.run(
        "PPO",
        config={**config, **search_space},
        storage_path="/home/rl/ray_results",
        stop={"timesteps_total": 500_000},
        num_samples=10,
        metric="env_runners/episode_return_mean",  # Cambiado a la métrica correcta
        mode="max",  # Maximizar la recompensa promedio
        checkpoint_freq=10,
        checkpoint_at_end=True,
    )

    print("Resultados del entrenamiento:")
    print(analysis.best_config)

if __name__ == "__main__":
    main()
