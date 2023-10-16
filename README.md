# Repositorio para la Asignatura de Data Science

Este repositorio contiene scripts y archivos asociados a la asignatura de Data Science. Los scripts se emplean para la limpieza, análisis y modelado de un dataset que se utiliza durante el curso.

Para los estudiantes UDP, se facilita el acceso al dataset, así como a los informes asociados, a través de [este enlace de Drive](https://drive.google.com/file/d/1be-HlC7Yh1PhP-F0AmuO0vyUqfcWr8cv/view?usp=sharing). En esta carpeta de Drive comprimida en zip, encontrarás organizados en carpetas tanto el dataset como los informes generados para su análisis (carpetas datasets y reports).

Para referencia adicional o para quienes deseen acceder a la fuente original, el dataset ha sido obtenido del portal de datos abiertos del Ministerio de Educación de Chile. Puedes acceder y descargar la versión 2022 del dataset directamente [aquí](https://datosabiertos.mineduc.cl/rendimiento-por-estudiante-2/).

## Estructura del Repositorio

Asegúrate de que las carpetas `datasets` y `reports` se encuentren en el directorio raíz del repositorio. La estructura actual del repositorio es la siguiente:

-   README.md
-   busqueda_grilla_hiperparametros.py
-   clean_dataset.py
-   correlacion_asistencia_rendimiento.py
-   data_analyst.py
-   gpu_detector.py
-   importancia_caracteristicas_random_forest.py
-   importancia_variables.py
-   random_forest_clean.py
-   random_forest_clean_mejores_hiperparam.py
-   red_neuronal.py
-   datasets/
    -   cleaned_data.csv
    -   20230209_Rendimiento_2022_20230131_WEB.csv
-   reports/
    -   report.html

## Archivos en este Repositorio

### 1. [busqueda_grilla_hiperparametros.py](./busqueda_grilla_hiperparametros.py)

Script para realizar una búsqueda en grilla de los mejores hiperparámetros para un modelo de Random Forest. [Ver código](./busqueda_grilla_hiperparametros.py).

### 2. [clean_dataset.py](./clean_dataset.py)

Este script realiza la limpieza y preprocesamiento inicial del dataset, eliminando datos innecesarios o incorrectos. [Ver código](./clean_dataset.py).

### 3. [correlacion_asistencia_rendimiento.py](./correlacion_asistencia_rendimiento.py)

Análisis de la correlación entre la asistencia y el rendimiento académico de los estudiantes. [Ver código](./correlacion_asistencia_rendimiento.py).

### 4. [data_analyst.py](./data_analyst.py)

Genera un informe de perfil del dataset para un análisis detallado de los datos. [Ver código](./data_analyst.py).

### 5. [gpu_detector.py](./gpu_detector.py)

Verifica y lista todos los dispositivos físicos disponibles, especialmente GPUs, y comprueba si TensorFlow ha sido compilado con soporte para CUDA. [Ver código](./gpu_detector.py).

### 6. [importancia_caracteristicas_random_forest.py](./importancia_caracteristicas_random_forest.py)

Este script visualiza la importancia de las características en un modelo de Random Forest. [Ver código](./importancia_caracteristicas_random_forest.py).

### 7. [importancia_variables.py](./importancia_variables.py)

Analiza y visualiza la importancia de las variables en el modelo de predicción. [Ver código](./importancia_variables.py).

### 8. [random_forest_clean.py](./random_forest_clean.py)

Implementa un modelo de Random Forest en los datos limpios y preprocesados. [Ver código](./random_forest_clean.py).

### 9. [random_forest_clean_mejores_hiperparam.py](./random_forest_clean_mejores_hiperparam.py)

Implementa un modelo de Random Forest con los mejores hiperparámetros encontrados a partir de la búsqueda en grilla. [Ver código](./random_forest_clean_mejores_hiperparam.py).

### 10. [red_neuronal.py](./red_neuronal.py)

Implementa una red neuronal para predecir el rendimiento académico de los estudiantes basado en sus características. [Ver código](./red_neuronal.py).

## Dataset

El dataset utilizado en estos scripts ha sido limpiado y preprocesado para facilitar el análisis y modelado. Se encuentra disponible en la carpeta `datasets`.

### Pre-requisitos

-   Asegúrate de tener [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado en tu sistema.

### Instalación del Entorno

1. **Clonar el Repositorio:**
   Clona este repositorio a tu máquina local para acceder al archivo de configuración del entorno Conda.

2. **Navegar a la Carpeta del Repositorio:**
   Utiliza la terminal o la línea de comandos para navegar a la carpeta donde se encuentra clonado el repositorio.

3. **Crear el Entorno Conda:**
   Ejecuta el siguiente comando para crear un nuevo entorno Conda a partir del archivo `environment.yml` proporcionado en el repositorio:

    ```bash
    conda env create -f environment.yml
    ```

    Esto instalará todas las dependencias necesarias en un nuevo entorno Conda.

4. **Activar el Entorno:**
   Una vez que la instalación esté completa, activa el entorno con el siguiente comando:

    ```bash
    conda activate myenv
    ```

    Asegúrate de reemplazar `myenv` con el nombre actual del entorno especificado en el archivo `environment.yml`.

### Uso del Entorno

Con el entorno Conda activado, ahora puedes ejecutar los scripts y notebooks del proyecto asegurando la consistencia en las dependencias y versiones de las librerías.

Para desactivar el entorno cuando hayas terminado, simplemente ejecuta:

```bash
conda deactivate

```

## Resultados

### Importancia de las Variables (`importancia_variables.py`)

Se utilizó el script [importancia_variables.py](./importancia_variables.py) para analizar la importancia relativa de las variables en el dataset. Los resultados, expresados en porcentajes, fueron los siguientes:

```plaintext
          Feature  Importance (%)
0      ASISTENCIA           48.48
5         COD_JOR           15.03
7  COD_DEPROV_RBD           10.67
3        COD_ENSE           10.31
1       COD_GRADO            6.14
4        COD_DEPE            4.86
6     COD_REG_RBD            3.46
2       RURAL_RBD            1.05
```

Resultados del Modelo Random Forest (random_forest_clean_mejores_hiperparam.py)
Se utilizaron los mejores hiperparámetros identificados para entrenar el modelo Random Forest, y se obtuvieron los siguientes resultados de error cuadrático medio (MSE), error absoluto medio (MAE) y coeficiente de determinación (R^2):

```plaintext
MSE de Entrenamiento: 0.25776162697809235
MSE de Prueba: 0.3562078175185358
MAE de Entrenamiento: 0.3550086595921462
MAE de Prueba: 0.4116183435956437
R^2 de Entrenamiento: 0.7137655783836987
R^2 de Prueba: 0.6075955473627808
```
