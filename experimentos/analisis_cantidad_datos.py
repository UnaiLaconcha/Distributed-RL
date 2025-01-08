import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Ruta al directorio de los logs de TensorBoard
logdir = "/home/ec2-user/mujoco/RAY/logs-7-batch-12000-/humanoid_ppo_timesteps_2_000_000"

# Métricas a extraer
metrics_to_extract = [
    "agent/reward_mean", "agent/reward_max", "agent/reward_min", "agent/episode_len_mean",
    "agent/kl_divergence", "training/total_loss", "training/policy_loss",
    "training/grad_norm", "training/learning_throughput", "system/cpu_utilization",
    "system/ram_utilization", "system/sampling_throughput", "system/sample_time_ms",
    "system/time_total_s", "model/value_explained_var"
]

# Crear un DataFrame vacío para almacenar los datos
data = []

# Recorrer cada archivo de eventos en el directorio
for subdir, dirs, files in os.walk(logdir):
    for file in files:
        if "events.out.tfevents" in file:
            event_file = os.path.join(subdir, file)

            # Cargar el acumulador de eventos
            ea = EventAccumulator(event_file)
            ea.Reload()

            # Extraer datos de las métricas especificadas
            for metric in metrics_to_extract:
                if metric in ea.Tags().get('scalars', []):
                    scalar_events = ea.Scalars(metric)
                    for event in scalar_events:
                        data.append({
                            "tag": metric,
                            "step": event.step,
                            "value": event.value,
                            "wall_time": event.wall_time
                        })

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Mostrar las primeras filas del DataFrame
print(df.head(), df.shape)