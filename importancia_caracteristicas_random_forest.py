import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import __version__ as sklearn_version

# Cambia la ruta del archivo según tu estructura de directorios
file_path = './datasets/20230209_Rendimiento_2022_20230131_WEB.csv'

print("Cargando el dataset...")
data = pd.read_csv(file_path, delimiter=';',
                   encoding='utf-8-sig', low_memory=False)

# Usar solo un subconjunto de los datos para una ejecución rápida
data = data.sample(frac=0.01, random_state=42)

# Preprocesamiento de datos
data = data.apply(lambda x: x.str.replace(
    ',', '.') if x.dtype == "object" else x)
data = data.apply(lambda x: pd.to_numeric(x, errors='ignore'))

# Dividir el dataset en conjuntos de entrenamiento y prueba
X = data.drop(columns=['PROM_GRAL'])
y = data['PROM_GRAL']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Identificar características numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Crear transformadores para características numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Aplicar ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Entrenar un modelo Random Forest
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

print("Entrenando el modelo...")
model.fit(X_train, y_train)

# Obtener la importancia de las características
importances = model.named_steps['classifier'].feature_importances_

# Obtener los nombres de las características después de la transformación
numeric_features_list = numeric_features.tolist()

# Verificar la versión de scikit-learn y obtener los nombres de las características transformadas
if sklearn_version >= '0.24':
    categorical_features_transformed = (model.named_steps['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(input_features=categorical_features))
else:
    categorical_features_transformed = (model.named_steps['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names(input_features=categorical_features))

all_features = numeric_features_list + categorical_features_transformed.tolist()

# Mostrar la importancia de las características en un gráfico
# Muestra las 10 características más importantes
indices = np.argsort(importances)[-10:]

plt.title('Importancia de las características')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [all_features[i] for i in indices])
plt.xlabel('Importancia relativa')
plt.show()
