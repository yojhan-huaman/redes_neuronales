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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Diccionario para mapear los síntomas con formato de codificación a un nombre más legible
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

# Función para generar el gráfico de torta (pie chart)
def generate_pie_chart(top_diseases, is_dark_mode=False):
    labels = []
    data = []
    total = 0

    for disease in top_diseases:
        labels.append(disease[0])
        data.append(disease[1])
        total += disease[1]

    other = 100 - total
    if other > 0.5:
        labels.append('Otras enfermedades')
        data.append(other)

    # Generar colores aleatorios
    def random_color():
        return f'#{random.randint(0, 0xFFFFFF):06x}'

    random_colors = [random_color() for _ in range(len(labels))]

    # Colores dinámicos según modo oscuro
    label_color = 'white' if is_dark_mode else 'black'
    title_color = 'white' if is_dark_mode else 'black'

    # Crear gráfico con fondo transparente
    fig = plt.figure(figsize=(8, 8), facecolor='none')
    sns.set(style="whitegrid")
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')  # Fondo transparente

    # Limpiar el gráfico anterior
    ax.clear()

    wedges, texts, autotexts = ax.pie(
        data,
        labels=labels,
        autopct='%1.1f%%',
        colors=random_colors,
        startangle=140
    )

    # Aplica correctamente los colores a las etiquetas y los porcentajes
    plt.setp(texts, color=label_color)       # Etiquetas
    plt.setp(autotexts, color=label_color)   # Porcentajes

    # Título con color correcto
    ax.set_title("Distribución de Probabilidades", color=title_color, fontsize=14, weight='bold')

    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)

    chart_data = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return chart_data

# Función para cargar los datos del modelo
def load_data():
    global model, symptoms, diseases, idx_to_disease, scaler
    if (os.path.exists('disease_classifier.pth') and 
        os.path.exists('symptoms.pkl') and 
        os.path.exists('diseases.pkl') and 
        os.path.exists('idx_to_disease.pkl') and
        os.path.exists('scaler.pkl')):

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

        print("Modelo y datos cargados correctamente")
    else:
        print("Error: Archivos del modelo no encontrados. Ejecute train.py primero.")
        raise FileNotFoundError("No se encontraron los archivos necesarios para el modelo. Por favor, ejecute train.py.")

@app.route('/', methods=['GET', 'HEAD'])
def index():
    if request.method == 'HEAD':
        return '', 200
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    print("Recibiendo solicitud de predicción...")

    # Obtener el valor de dark_mode desde el formulario
    dark_mode = request.form.get('dark_mode') == 'true'
    print(f"Modo oscuro activado: {dark_mode}")  # Verificar si se ha recibido correctamente

    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        return render_template('index.html', symptoms=symptoms, error="Por favor seleccione al menos un síntoma")

    translated_symptoms = [symptom_translation.get(symptom, symptom) for symptom in selected_symptoms]

    # Realizar la predicción
    predicted_disease, probabilities = predict_disease(
        model, selected_symptoms, symptoms, scaler, idx_to_disease
    )
    
    # Obtener las 3 enfermedades más probables
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_diseases = [(idx_to_disease[idx], float(probabilities[idx]) * 100) for idx in top_indices]
    
    # Generar el gráfico de torta
    chart_data = generate_pie_chart(top_diseases, is_dark_mode=dark_mode)

    print(f"Predicción: {predicted_disease}")
    print(f"Top enfermedades: {top_diseases}")

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
    # Obtener el valor de dark_mode desde el formulario (AJAX)
    dark_mode = request.form.get('dark_mode') == 'true'
    print(f"Modo oscuro activado: {dark_mode}")  # Verificar si se ha recibido correctamente

    # Aquí seleccionamos algunos síntomas para generar el gráfico (puedes modificar esta parte)
    selected_symptoms = ['fiebre', 'tos', 'dolor_de_cabeza']  # Ejemplo de síntomas

    # Realizar la predicción (puedes modificarlo según tu lógica)
    predicted_disease, probabilities = predict_disease(
        model, selected_symptoms, symptoms, scaler, idx_to_disease
    )
    
    # Obtener las 3 enfermedades más probables
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_diseases = [(idx_to_disease[idx], float(probabilities[idx]) * 100) for idx in top_indices]
    
    # Generar el gráfico de torta
    chart_data = generate_pie_chart(top_diseases, is_dark_mode=dark_mode)

    # Retornar el gráfico como Base64
    return {"chart_data": chart_data}
load_data()
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
