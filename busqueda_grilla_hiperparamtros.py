import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Cargar los datos limpios
cleaned_data_path = './datasets/cleaned_data.csv'
cleaned_df = pd.read_csv(cleaned_data_path)

# Seleccionar las características para el modelado
selected_features = [
    'ASISTENCIA', 'COD_GRADO', 'RURAL_RBD', 'COD_ENSE',
    'COD_DEPE', 'COD_JOR', 'COD_REG_RBD', 'COD_DEPROV_RBD', 'PROM_GRAL'
]

# Crear un subset de datos con las características seleccionadas
modeling_data = cleaned_df[selected_features]

# Codificar variables categóricas
le = LabelEncoder()
for col in modeling_data.select_dtypes(include=['object', 'category']).columns:
    modeling_data[col] = le.fit_transform(modeling_data[col].astype(str))

# Dividir los datos en conjuntos de entrenamiento y prueba
X = modeling_data.drop(columns=['PROM_GRAL'])
y = modeling_data['PROM_GRAL']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Definir los hiperparámetros y sus valores para la búsqueda en grilla
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear el modelo Random Forest
rf = RandomForestRegressor(random_state=42)

# Instanciar la búsqueda en grilla
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')

# Ajustar la búsqueda en grilla a los datos
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros encontrados en la búsqueda
best_params = grid_search.best_params_

print("Mejores hiperparámetros encontrados:")
print(best_params)
