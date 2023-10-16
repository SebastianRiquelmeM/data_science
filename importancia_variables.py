import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
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

# Entrenar el modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Obtener la importancia de las características
feature_importances = rf_model.feature_importances_

# Crear un DataFrame para visualizar la importancia de las características
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Ordenar las características por importancia
features_df = features_df.sort_values(by='Importance', ascending=False)

# Imprimir la importancia de las características
print(features_df)

# Graficar la importancia de las características
plt.figure(figsize=(10, 6))
plt.bar(features_df['Feature'], features_df['Importance'], color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()
