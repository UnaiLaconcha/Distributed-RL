from pyspark.sql import SparkSession
from train import convert
import os

# Crear la sesión de Spark
spark = SparkSession.builder \
    .appName("Leer datos particionados") \
    .getOrCreate()

for folder_name in os.listdir('RAY'):
    folder_path = os.path.join('RAY',folder_name)
    file_name = os.listdir(folder_path)[0]
    log_dir = os.path.join(folder_path,file_name)
    save_dir = 'hdfs://hadoop-master:9000/home/ec2-user/RAY/metrics'
    convert(log_dir, save_dir, spark)

# Detener la sesión de Spark
spark.stop()
