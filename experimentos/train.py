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
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, FloatType

def agent_metrics(results, writer, total_steps):
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

def train_metrics(results, writer, total_steps):
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

def system_metrics(results, writer, total_steps):
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

def model_metrics(results, writer, total_steps):
    # ==========================================================
    # MÉTRICAS DE CALIDAD DEL MODELO
    # ==========================================================

    # 16. Valor explicado por la función de valor
    value_explained_var = results['info']['learner']['default_policy']['learner_stats'].get('vf_explained_var', None)
    if value_explained_var is not None:
        writer.add_scalar('model/value_explained_var', value_explained_var, total_steps)

def create_env(env_config):
    return gym.make(env_name)


def convert(log_dir, save_dir, spark):
    for path in os.listdir(log_dir):
        if "events.out.tfevents" in path:
            # Ruta al archivo de eventos
            event_file = os.path.join(log_dir,path)

    father_path = os.path.basename(os.path.dirname(log_dir))
    print("\n\n Otro path",father_path,"\n\n")

    output_csv = os.path.join(save_dir,father_path)

    # Cargar el archivo de eventos
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    metrics_to_extract = [
        "agent/reward_mean", "agent/reward_max", "agent/reward_min", "agent/episode_len_mean",
        "agent/kl_divergence", "training/total_loss", "training/policy_loss",
        "training/grad_norm", "training/learning_throughput", "system/cpu_utilization",
        "system/ram_utilization", "system/sampling_throughput", "system/sample_time_ms",
        "system/time_total_s", "model/value_explained_var"
    ]

    # Crear un diccionario para almacenar las métricas
    metrics_data = {metric: [] for metric in metrics_to_extract}  # Diccionario para almacenar eventos por métrica

    # Extraer datos de TensorBoard
    for metric in metrics_to_extract:
        if metric in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(metric)
            for event in scalar_events:
                step = event.step
                value = event.value
                wall_time = event.wall_time
                metrics_data[metric].append({"step": step, "value": value, "wall_time": wall_time})

    # Transformar el diccionario de métricas en una lista de filas para Spark
    rows = []
    for metric, events in metrics_data.items():
        for event in events:
            rows.append({"tag": metric, "step": event["step"], "value": event["value"], "wall_time": event["wall_time"]})

    # Definir el esquema para el DataFrame
    schema = StructType([
        StructField("tag", StringType(), True),
        StructField("step", IntegerType(), True),
        StructField("value", FloatType(), True),
        StructField("wall_time", FloatType(), True)
    ])

    # Crear el DataFrame de Spark directamente
    spark_df = spark.createDataFrame(rows, schema=schema)

    # primeros datos
    spark_df.show()
    print("\n\n\n ",output_csv, '\n\n\n')
    spark_df.write.mode("overwrite").csv(output_csv, header=True)
    
    print(f"DataFrame guardado exitosamente en HDFS en {output_csv}")


def main(env_name, model_name, hyperparams, log_dir, workers, cpus_task,
            cpus_worker, memory, env_variable, timesteps, train_batch_size, sgd_minibatch_size,
            gamma, lr, model, seed, save_dir):
    try:
        # Configuración de Spark
        spark = SparkSession \
            .builder \
            .appName("Ray on Spark Example") \
            .config("spark.task.cpus", cpus_task)  \
            .config("spark.executor.cores", cpus)  \
            .config("spark.num.executors", workers)  \
            .config("spark.executor.memory", memory)  \
            .config("spark.executorEnv.PATH", env_variable) \
            .getOrCreate()

        # Configuración de Ray con Spark
        setup_ray_cluster(max_worker_nodes=workers, num_cpus_worker_node=cpus, num_gpus_worker_node=0)

        # Inicializar Ray
        ray.init(ignore_reinit_error=True,
                )

        resourses = ray.cluster_resources()

        print("===== Recursos disponibles en el cluster de Ray =====")
        print(json.dumps(resourses, indent=4, sort_keys=True))
        print("=====================================================")

        register_env(env_name, create_env)

        # Crear un escritor de TensorBoard
        writer = SummaryWriter(log_dir)

        # Configurar y entrenar el modelo
        config = {
            "env": env_name,
            "num_workers": workers - 1,
            "framework": "torch",
            "logger_config": {
                "loggers": [
                    JsonLoggerCallback,
                    CSVLoggerCallback,
                    TBXLoggerCallback  # Asegúrate de que TBXLoggerCallback está incluido
                ]
            },
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "gamma":gamma,
            "lr": lr,
            "model": model,
            "api_stack": {"enable_rl_module_and_learner": False, "enable_env_runner_and_connector_v2": False},  # Deshabilitar la nueva API
            "seed": seed
        }

        algo = PPO(config=config)

        # Entrenamiento
        total_steps = 0

        while total_steps < timesteps:
            results = algo.train()
            total_steps = results.get("timesteps_total", results.get("timesteps_this_iter", 0))

            # Metricas del agente
            agent_metrics(results, writer, total_steps)

            # Metricas de entrenamiento
            train_metrics(results, writer, total_steps)

            # Metricas del sistema
            system_metrics(results, writer, total_steps)

            # Metricas del modelo
            model_metrics(results, writer, total_steps)

        # Cerrar el escritor de TensorBoard
        writer.close()

        # Guardar el modelo
        checkpoint_dir = os.path.join(log_dir, f"{model_name}_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        algo.save_checkpoint(checkpoint_dir)

        print(f"Model and logs saved at: {log_dir}")

        algo.stop()
        
        shutdown_ray_cluster()

        # Convertir formato de salida de tensorboard
        convert(log_dir, save_dir, spark)

    except Exception as err: 
        print("Deteniendo cluster por el siguiente error: ",err)
    
        shutdown_ray_cluster()

if __name__ == "__main__":
    # Configuracion parametros del cluster
    workers = 7
    cpus_task = 2
    cpus = 2
    cpus_worker = 2
    memory = '4g'
    env_variable = "/home/ec2-user/spark-3.5.3-bin-hadoop3/bin:/home/ec2-user/hadoop-3.3.6/bin:/home/ec2-user/spark-3.5.3-bin-hadoop3/bin:/home/ec2-user/hadoop-3.3.6/bin:/home/ec2-user/.local/bin:/home/ec2-user/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

    # Configuración de parámetros del ambiente
    env_name = "Humanoid-v4"
    model_name = "humanoid_ppo"
    hyperparams = "timesteps_200k_lr_0.0003"
    log_dir = f"RAY/logs-prueba-batch-12000/{model_name}_{hyperparams}"
    save_dir = 'hdfs://hadoop-master:9000/home/ec2-user/RAY/metrics'

    os.makedirs(log_dir, exist_ok=True)

    # Parametros de entrenamiento
    timesteps = 600
    train_batch_size = 200
    sgd_minibatch_size = 128
    gamma = 0.9537564860944372
    lr = 5.70604722100944e-05
    model = {
        "fcnet_hiddens": [128, 128],           
        "fcnet_activation": "tanh"   
        }

    # Semilla para reproducibilidad
    seed = 42
    
    main(env_name, model_name, hyperparams, log_dir, workers, cpus_task,
            cpus_worker, memory, env_variable, timesteps, train_batch_size, sgd_minibatch_size,
            gamma, lr, model, seed, save_dir)