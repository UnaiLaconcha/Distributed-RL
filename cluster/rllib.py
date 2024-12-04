import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from pyspark.sql import SparkSession

# Inicializa SparkSession
spark = SparkSession.builder \
    .appName("CartPole-RL-RLLib") \
    .getOrCreate()

# Inicializa Ray en modo Spark
ray.init(address="spark://master:7077")  # Conéctate a un clúster Ray existente o inicia uno en Spark

# Configuración de PPO para CartPole
config = (
    PPOConfig()
    .environment("CartPole-v1")  # Nombre del entorno
    .rollouts(num_rollout_workers=2)  # Número de workers para recolectar datos
    .framework("torch")  # Usar PyTorch (puedes cambiar a TensorFlow si lo prefieres)
)

# Define una función de entrenamiento
def train_cartpole(config, num_iterations=200000):
    # Entrenamiento con Tune
    tune.run(
        "PPO",  # Algoritmo a usar (en este caso, PPO)
        config=config.to_dict(),
        stop={"episode_reward_mean": 300},  # Detenerse cuando se alcanza una recompensa media de 200
        local_dir="./ray_results",  # Carpeta donde guardar resultados
        verbose=1,  # Nivel de detalle de logs
        num_samples=1,  # Número de configuraciones de prueba
    )

# Ejecuta el entrenamiento
train_cartpole(config)

# Cierra Ray y Spark al final
ray.shutdown()
spark.stop()

