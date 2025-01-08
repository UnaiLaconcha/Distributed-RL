import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Ruta al archivo de eventos
event_file = "/home/rl/proyectos/proyRL/RAY_humanoid/logs/dqn_Ts200kLr0.0003/events.out.tfevents.1735213687.Unai.1200499.0"
output_csv = "/home/rl/proyectos/proyRL/RAY_humanoid/logs/dqn_Ts200kLr0.0003/dqn_Ts200kLr0.0003.csv"

# Cargar el archivo de eventos
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# Métricas que queremos extraer
metrics_to_extract = [
    "agent/reward_mean", "agent/reward_max", "agent/reward_min", "agent/episode_len_mean",
    "agent/kl_divergence", "training/total_loss", "training/policy_loss",
    "training/grad_norm", "training/learning_throughput", "system/cpu_utilization",
    "system/ram_utilization", "system/sampling_throughput", "system/sample_time_ms",
    "system/time_total_s", "model/value_explained_var"
]

# Crear un diccionario para almacenar las métricas
metrics_data = {metric: {} for metric in metrics_to_extract}  # Diccionario anidado para steps y valores

# Extraer datos de TensorBoard
for metric in metrics_to_extract:
    if metric in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(metric)
        for event in scalar_events:
            step = event.step
            value = event.value
            if step not in metrics_data[metric]:
                metrics_data[metric][step] = value

# Construir el DataFrame con steps como índice
df = pd.DataFrame({metric: pd.Series(data) for metric, data in metrics_data.items()})

# Asegurar que el índice sea los pasos (steps)
df.index.name = "steps"
df = df.sort_index()  # Ordenar por índice para consistencia
print(df.head())

# Exportar a CSV con steps como la primera columna
df.to_csv(output_csv)

print(f"DataFrame creado y exportado a {output_csv}")
