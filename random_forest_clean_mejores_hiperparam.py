import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# Entrenar el modelo Random Forest con los hiperparámetros optimizados
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1  # Utilizar todos los núcleos de CPU disponibles
)
rf_model.fit(X_train, y_train)

# Hacer predicciones
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calcular métricas
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Imprimir métricas
print(f'MSE de Entrenamiento: {mse_train}')
print(f'MSE de Prueba: {mse_test}')
print(f'MAE de Entrenamiento: {mae_train}')
print(f'MAE de Prueba: {mae_test}')
print(f'R^2 de Entrenamiento: {r2_train}')
print(f'R^2 de Prueba: {r2_test}')

# Visualizar la importancia de las características
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel('Importancia de la Característica')
plt.title('Visualización de la Importancia de las Características')
plt.show()
