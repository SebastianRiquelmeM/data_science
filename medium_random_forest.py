import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Cambia la ruta del archivo según tu estructura de directorios
file_path = './datasets/20230209_Rendimiento_2022_20230131_WEB.csv'

print("Cargando el dataset...")
data = pd.read_csv(file_path, delimiter=';',
                   encoding='utf-8-sig', low_memory=False)

# Tomando una muestra del 10% del dataset para acelerar el entrenamiento
data = data.sample(frac=0.1, random_state=42)

# Aplicar la operación de string solo donde es aplicable
data['PROM_GRAL'] = data['PROM_GRAL'].apply(lambda x: str(
    x).replace(',', '.') if isinstance(x, str) else x).astype(float)
data['ASISTENCIA'] = data['ASISTENCIA'].apply(lambda x: str(
    x).replace(',', '.') if isinstance(x, str) else x).astype(float)

print("Seleccionando características y target...")
features = ['GEN_ALU', 'COD_ENSE', 'COD_GRADO', 'RBD', 'NOM_RBD', 'ASISTENCIA']
target = 'PROM_GRAL'

X = data[features]
y = data[target]

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocesamiento
numeric_features = ['ASISTENCIA']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['GEN_ALU', 'COD_ENSE', 'COD_GRADO', 'RBD', 'NOM_RBD']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Reduciendo el número de estimadores para acelerar el entrenamiento
# rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', rf)
                           ])

print("Entrenando el modelo...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

importances = pipeline.named_steps['model'].feature_importances_
print("Importancia de las características:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")
