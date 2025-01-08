# Distributed-RL

Este proyecto implementa un entorno de aprendizaje por refuerzo distribuido utilizando Ray y RLlib en un clúster de Hadoop.

## Configuración del Clúster

Para crear y configurar el clúster de Hadoop, utiliza el script de bash main.sh. Este script automatiza el proceso de configuración del entorno distribuido. Para ejecutarlo, usa el siguiente comando:

###
./main.sh
###

Este script realizará las siguientes tareas:
1. Crear instancias EC2 en AWS
2. Configurar Hadoop y Spark en estas instancias
3. Establecer la red y los grupos de seguridad necesarios
4. Inicializar el clúster de Ray para la computación distribuida

## Experimentos

La carpeta experimentos contiene scripts de Python para ejecutar varios experimentos de aprendizaje por refuerzo utilizando RLlib. Estos scripts demuestran cómo:

- Entrenar modelos de RL de manera distribuida
- Cargar y procesar datos en HDFS
- Analizar el rendimiento de diferentes algoritmos de RL

## Análisis

Para un análisis en profundidad y visualización de los resultados de entrenamiento y el rendimiento de la computación distribuida, consulta el notebook Jupyter ray_analisis.ipynb. Este notebook incluye:

- Consultas para extraer métricas de rendimiento de los registros de entrenamiento
- Visualizaciones para comparar diferentes ejecuciones de entrenamiento
- Análisis de la escalabilidad y eficiencia del enfoque distribuido