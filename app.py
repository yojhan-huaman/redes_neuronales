from flask import Flask, render_template, request
import os
import random
import torch
import pickle
import numpy as np
from model import DiseaseClassifier, load_model, predict_disease
import matplotlib
matplotlib.use('Agg')  # Usa el backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Diccionario para mapear los síntomas con nombres legibles
symptom_translation = {
    'dolor_muscular': 'Dolor muscular',
    'dolor_de_garganta': 'Dolor de garganta',
    'dificultad_para_respirar': 'Dificultad para respirar',
    'tos': 'Tos',
    'fiebre': 'Fiebre',
    'cansancio': 'Cansancio',
    'dolor_de_estomago': 'Dolor de estómago',
    'náuseas': 'Náuseas',
    'vómitos': 'Vómitos',
    'dolor_abdominal': 'Dolor abdominal',
    'escurrimiento_nasal': 'Escurrimiento nasal',
    'dolor_de_cabeza': 'Dolor de cabeza',
    'perdida_del_apetito': 'Pérdida del apetito',
    'diarrea': 'Diarrea',
}

# Variables globales
model = None
symptoms = None
diseases = None
idx_to_disease = None
scaler = None

def generate_pie_chart(top_diseases, is_dark_mode=False):
    labels = [d[0] for d in top_diseases]
    data = [d[1] for d in top_diseases]
    total = sum(data)

    other = 100 - total
    if other > 0.5:
        labels.append('Otras enfermedades')
        data.append(other)

    random_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(labels))]
    label_color = 'white' if is_dark_mode else 'black'
    title_color = 'white' if is_dark_mode else 'black'

    fig = plt.figure(figsize=(8, 8), facecolor='none')
    sns.set(style="whitegrid")
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.clear()

    wedges, texts, autotexts = ax.pie(
        data,
        labels=labels,
        autopct='%1.1f%%',
        colors=random_colors,
        startangle=140
    )

    plt.setp(texts, color=label_color)
    plt.setp(autotexts, color=label_color)
    ax.set_title("Distribución de Probabilidades", color=title_color, fontsize=14, weight='bold')

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)

    chart_data = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return chart_data

def load_data():
    global model, symptoms, diseases, idx_to_disease, scaler
    print("Cargando archivos del modelo...")
    try:
        with open('symptoms.pkl', 'rb') as f:
            symptoms = pickle.load(f)
        with open('diseases.pkl', 'rb') as f:
            diseases = pickle.load(f)
        with open('idx_to_disease.pkl', 'rb') as f:
            idx_to_disease = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        input_size = len(symptoms)
        hidden_size = 128
        num_classes = len(diseases)
        model = load_model(input_size, hidden_size, num_classes)

        print("Modelo y datos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar los datos del modelo: {e}")
        raise

@app.route('/', methods=['GET', 'HEAD'])
def index():
    global symptoms
    if request.method == 'HEAD':
        return '', 200

    if symptoms is None:
        return "Error: los datos del modelo no se cargaron correctamente.", 500

    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    global model, symptoms, scaler, idx_to_disease

    if symptoms is None:
        return "Error: los datos del modelo no se cargaron correctamente.", 500

    print("Recibiendo solicitud de predicción...")
    dark_mode = request.form.get('dark_mode') == 'true'
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        return render_template('index.html', symptoms=symptoms, error="Por favor seleccione al menos un síntoma")

    translated_symptoms = [symptom_translation.get(symptom, symptom) for symptom in selected_symptoms]

    predicted_disease, probabilities = predict_disease(
        model, selected_symptoms, symptoms, scaler, idx_to_disease
    )

    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_diseases = [(idx_to_disease[idx], float(probabilities[idx]) * 100) for idx in top_indices]

    chart_data = generate_pie_chart(top_diseases, is_dark_mode=dark_mode)

    return render_template(
        'index.html',
        symptoms=symptoms,
        prediction=predicted_disease,
        top_diseases=top_diseases,
        selected_symptoms=translated_symptoms,
        chart_data=chart_data
    )

@app.route('/update_chart', methods=['POST'])
def update_chart():
    global model, symptoms, scaler, idx_to_disease
    dark_mode = request.form.get('dark_mode') == 'true'
    selected_symptoms = ['fiebre', 'tos', 'dolor_de_cabeza']

    predicted_disease, probabilities = predict_disease(
        model, selected_symptoms, symptoms, scaler, idx_to_disease
    )

    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_diseases = [(idx_to_disease[idx], float(probabilities[idx]) * 100) for idx in top_indices]

    chart_data = generate_pie_chart(top_diseases, is_dark_mode=dark_mode)
    return {"chart_data": chart_data}

# Cargar datos antes de levantar la app
try:
    load_data()
except Exception as e:
    print(f"No se pudo iniciar la app por error en load_data: {e}")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
