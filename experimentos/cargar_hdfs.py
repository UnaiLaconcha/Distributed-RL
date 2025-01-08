from pyspark.sql import SparkSession

# Crear la sesión de Spark
spark = SparkSession.builder \
    .appName("Leer datos particionados") \
    .getOrCreate()

# Ruta al directorio donde están los archivos particionados
hdfs_path = "hdfs://hadoop-master:9000/home/ec2-user/RAY/metrics/humanoid_ppo_timesteps_200k_lr_0.0003/"

# Leer los datos como un DataFrame
df = spark.read.csv(
    hdfs_path,
    header=True,  # Si los archivos tienen encabezados
    inferSchema=True  # Inferir automáticamente el tipo de datos
)

# Mostrar las primeras filas del DataFrame
df.show()

# Imprimir el esquema del DataFrame
df.printSchema()

# Detener la sesión de Spark
spark.stop()
