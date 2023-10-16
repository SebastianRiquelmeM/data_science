# Repositorio para la Asignatura de Data Science

Este repositorio contiene scripts y archivos asociados a la asignatura de Data Science. Los scripts son empleados para el análisis y modelado de un dataset que se utiliza durante el curso.

Para los estudiantes UDP, se facilita el acceso al dataset, así como a los informes asociados, a través de [este enlace de Drive](https://drive.google.com/file/d/1be-HlC7Yh1PhP-F0AmuO0vyUqfcWr8cv/view?usp=sharing). En esta carpeta de Drive comprimida en zip, encontrarás organizados en carpetas tanto el dataset como los informes generados para su análisis (carpetas datasets y reports).

Para referencia adicional o para quienes deseen acceder a la fuente original, el dataset ha sido obtenido del portal de datos abiertos del Ministerio de Educación de Chile. Puedes acceder y descargar la versión 2022 del dataset directamente [aquí](https://datosabiertos.mineduc.cl/rendimiento-por-estudiante-2/).

## Archivos en este Repositorio

### 1. [correlacion_asistencia_rendimiento.py](./correlacion_asistencia_rendimiento.py)

Este script está diseñado para analizar la correlación entre la asistencia y el rendimiento académico de los estudiantes. Utiliza matplotlib para visualizar la correlación en un scatter plot. [Ver código](./correlacion_asistencia_rendimiento.py).

### 2. [data_analyst.py](./data_analyst.py)

Genera un informe de perfil del dataset usando ydata_profiling, proporcionando un análisis exhaustivo del dataset para una inspección detallada. El informe se guarda en formato HTML. [Ver código](./data_analyst.py).

### 3. [gpu_detector.py](./gpu_detector.py)

Este script verifica y lista todos los dispositivos físicos disponibles, en particular GPUs, y comprueba si TensorFlow ha sido compilado con soporte para CUDA. [Ver código](./gpu_detector.py).

### 4. [medium_random_forest.py](./medium_random_forest.py)

Implementa un modelo de Random Forest utilizando una muestra del 10% del dataset para predecir el rendimiento académico basado en diversas características de los estudiantes. [Ver código](./medium_random_forest.py).

### 5. [min_random_forest.py](./min_random_forest.py)

Similar al script `medium_random_forest.py`, pero utiliza una muestra más pequeña del dataset para una ejecución más rápida, ideal para pruebas iniciales y depuración. [Ver código](./min_random_forest.py).

### 6. [random_forest.py](./random_forest.py)

Este es un script más robusto para implementar el modelo de Random Forest, utilizando todos los núcleos disponibles y un mayor número de estimadores para un análisis más profundo. [Ver código](./random_forest.py).

### 7. [red_neuronal.py](./red_neuronal.py)

Implementa una red neuronal para predecir el rendimiento académico de los estudiantes basado en sus características. Utiliza TensorFlow para construir, entrenar y evaluar el modelo. [Ver código](./red_neuronal.py).

## Dataset

El dataset utilizado en estos scripts se ha obtenido del sitio web del gobierno. Es una colección comprensiva de datos relacionados con el rendimiento académico de los estudiantes, incluyendo variables como asistencia, notas, y otros factores que podrían influir en el rendimiento académico.
