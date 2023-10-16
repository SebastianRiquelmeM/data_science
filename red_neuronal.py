import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf

# Cambia la ruta del archivo según tu estructura de directorios
file_path = './datasets/20230209_Rendimiento_2022_20230131_WEB.csv'

print("Cargando el dataset...")
data = pd.read_csv(file_path, delimiter=';',
                   encoding='utf-8-sig', low_memory=False)

print("Verificando y manejando valores NaN e infinitos...")
data['PROM_GRAL'] = pd.to_numeric(
    data['PROM_GRAL'].str.replace(',', '.'), errors='coerce')
data['ASISTENCIA'] = pd.to_numeric(
    data['ASISTENCIA'].str.replace(',', '.'), errors='coerce')

print("Seleccionando características y target...")
features = ['GEN_ALU', 'COD_ENSE', 'COD_GRADO', 'RBD', 'NOM_RBD', 'ASISTENCIA']
target = 'PROM_GRAL'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocesamiento
numeric_features = ['ASISTENCIA']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['GEN_ALU', 'COD_ENSE', 'COD_GRADO', 'RBD', 'NOM_RBD']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Aplicando preprocesamiento...")
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Crear y entrenar modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("Entrenando el modelo...")
model.fit(X_train, y_train, epochs=50, batch_size=1024, validation_split=0.1)

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error en el conjunto de prueba: {loss}")
