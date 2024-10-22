import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv)

def preprocesar_datos(df):
    # Aquí debes realizar el preprocesamiento necesario según tu dataset
    # Por ejemplo, convertir columnas categóricas a numéricas
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('aprobacion', axis=1)  # Suponiendo que aprobacion es la columna objetivo
    y = df['aprobacion']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def construir_modelo(input_shape):
    modelo = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

def entrenar_modelo(modelo, X_train, y_train):
    modelo.fit(X_train, y_train, epochs=50, batch_size=10)
    modelo.save('modelo_prestamo.h5')

