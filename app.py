from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Carga el modelo entrenado
model = joblib.load('random_forest_model.pkl')
app.logger.debug('Random Forest cargado correctamente')

# Mapeo de DESIGNATION (igual al entrenamiento)
designation_map = {
    'Data Engineer': 0,
    'Data Scientist': 1,
    'Machine Learning Engineer': 2,
    'Business Analyst': 3
}

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibe JSON y parsea
        data = request.get_json(force=True)
        
        designation = data.get('designation')
        tenure     = float(data.get('tenure', 0))
        age        = float(data.get('age', 0))
        exp        = float(data.get('exp', 0))
        leaves     = float(data.get('leaves', 0))
        
        # Valida y convierte DESIGNATION
        if designation not in designation_map:
            raise ValueError(f"Designación desconocida: {designation}")
        desig_num = designation_map[designation]

        # Construye DataFrame con el mismo orden de columnas
        df = pd.DataFrame([[desig_num, tenure, age, exp, leaves]],
                          columns=['DESIGNATION','TenureDays','AGE','PAST EXP','LEAVES USED'])
        app.logger.debug(f'DataFrame para predecir:\n{df}')

        # Predicción
        salary = model.predict(df)[0]
        return jsonify({'salary': round(float(salary), 2)})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)