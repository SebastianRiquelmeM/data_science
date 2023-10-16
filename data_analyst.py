# Importar bibliotecas necesarias
import pandas as pd
from ydata_profiling import profile_report

# Definir la ruta del archivo en tu máquina local (ajusta esta ruta según donde tengas guardado el archivo)
file_path = './datasets/20230209_Rendimiento_2022_20230131_WEB.csv'

# Cargar el archivo CSV usando el delimitador correcto y la codificación 'UTF-8-sig'
data = pd.read_csv(file_path, delimiter=';',
                   encoding='UTF-8-sig', low_memory=False)

# Generar informe de perfil usando ydata_profiling
report = data.profile_report(title='Report', progress_bar=True)

# Guardar el informe en tu máquina local (ajusta la ruta de salida según donde desees guardar el informe)
output_path = './reports/report.html'
report.to_file(output_path)
