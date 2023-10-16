import pandas as pd
import matplotlib.pyplot as plt

# Definir la ruta del archivo
file_path = './datasets/20230209_Rendimiento_2022_20230131_WEB.csv'

# Cargar el dataset
data = pd.read_csv(file_path, delimiter=';',
                   encoding='UTF-8-sig', low_memory=False)

# Tomar una muestra del 10% del dataset (puedes ajustar este porcentaje según tus necesidades)
sample_data = data.sample(frac=0.10)

# Convertir las columnas 'ASISTENCIA' y 'PROM_GRAL' a tipo float
sample_data['ASISTENCIA'] = pd.to_numeric(
    sample_data['ASISTENCIA'], errors='coerce')
sample_data['PROM_GRAL'] = pd.to_numeric(
    sample_data['PROM_GRAL'], errors='coerce')

# Descartar los valores NaN que pueden haber surgido al convertir a numérico
valid_data = sample_data.dropna(subset=['ASISTENCIA', 'PROM_GRAL'])

# Calcular coeficiente de correlación de Pearson
correlation = valid_data['ASISTENCIA'].corr(valid_data['PROM_GRAL'])

print(
    f"Correlación entre Asistencia y Desempeño Académico (basado en muestra): {correlation:.2f}")

# Graficar la relación con un scatter plot
plt.scatter(valid_data['ASISTENCIA'], valid_data['PROM_GRAL'], alpha=0.1)
plt.title("Relación entre Asistencia y Desempeño Académico (muestra)")
plt.xlabel("Asistencia")
plt.ylabel("Desempeño Académico (PROM_GRAL)")
plt.show()
