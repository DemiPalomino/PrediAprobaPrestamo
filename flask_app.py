from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from model import cargar_datos, preprocesar_datos

app = Flask(__name__)

# Cargar el modelo previamente entrenado
modelo = load_model('modelo_prestamo.h5')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    ingresos = float(request.form['ingresos'])
    historial_crediticio = float(request.form['historial_crediticio'])
    estado_civil = float(request.form['estado_civil'])
    empleo = float(request.form['empleo'])
    monto_solicitado = float(request.form['monto_solicitado'])
    plazo_prestamo = float(request.form['plazo_prestamo'])
    tasa_interes = float(request.form['tasa_interes'])

    # Preparar entrada para la predicción
    input_data = np.array([[ingresos, historial_crediticio, estado_civil, empleo,
                            monto_solicitado, plazo_prestamo, tasa_interes]])

    prediccion = modelo.predict(input_data)
    resultado = "Aprobado" if prediccion[0][0] > 0.5 else "No Aprobado"

    # Renderizar la nueva página con el resultado
    return render_template('resultado.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
