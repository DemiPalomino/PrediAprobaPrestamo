from model import cargar_datos, preprocesar_datos, construir_modelo, entrenar_modelo

# Cargar y preprocesar los datos
data = cargar_datos('dataset.csv')
X_train, X_test, y_train, y_test = preprocesar_datos(data)

# Construir el modelo
input_shape = (X_train.shape[1],)  # número de características
modelo = construir_modelo(input_shape)

# Entrenar el modelo
entrenar_modelo(modelo, X_train, y_train)
