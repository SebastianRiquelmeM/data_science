import pandas as pd

# Ruta del archivo
file_path = './datasets/20230209_Rendimiento_2022_20230131_WEB.csv'


def clean_and_preprocess(df):
    # Convertir columnas a tipos de datos adecuados
    df['PROM_GRAL'] = pd.to_numeric(df['PROM_GRAL'], errors='coerce')
    df['ASISTENCIA'] = pd.to_numeric(df['ASISTENCIA'], errors='coerce')

    # Eliminar filas con 'PROM_GRAL' y 'ASISTENCIA' igual a cero
    cleaned_df = df[(df['PROM_GRAL'] > 0) & (df['ASISTENCIA'] > 0)]

    # Eliminar filas con 'PROM_GRAL' mayor a 7 y 'ASISTENCIA' mayor a 100%
    cleaned_df = cleaned_df[(cleaned_df['PROM_GRAL'] <= 7) & (
        cleaned_df['ASISTENCIA'] <= 100)]

    # Eliminar filas con valores faltantes
    cleaned_df = cleaned_df.dropna()

    return cleaned_df


# Leer el dataset completo
df = pd.read_csv(file_path, sep=';', encoding='latin1')

# Limpiar el dataset
cleaned_df = clean_and_preprocess(df)

# Guardar el DataFrame limpio en un nuevo archivo CSV
cleaned_df.to_csv('./datasets/cleaned_data.csv', index=False)

print("La data limpiada se ha guardado en './datasets/cleaned_data.csv'")
