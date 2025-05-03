from flask import Flask, render_template, request
import os
import random
import torch
import pickle
import numpy as np
import traceback
import sys
import logging
# Import the rest only if needed to avoid failures
try:
    from model import DiseaseClassifier, load_model, predict_disease
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global variables
model = None
symptoms = ['tos', 'fiebre', 'cansancio', 'dolor_de_cabeza', 'dolor_muscular', 
           'dolor_de_garganta', 'dificultad_para_respirar', 'dolor_de_estomago',
           'náuseas', 'vómitos', 'dolor_abdominal', 'escurrimiento_nasal',
           'perdida_del_apetito', 'diarrea']
diseases = None
idx_to_disease = None
scaler = None

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

def generate_pie_chart(top_diseases, is_dark_mode=False):
    try:
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
    except Exception as e:
        logger.error(f"Error generating pie chart: {e}")
        logger.error(traceback.format_exc())
        return None

def load_data():
    global model, symptoms, diseases, idx_to_disease, scaler
    
    logger.info("Attempting to load model data files...")
    
    try:
        # Check if pickle files exist
        required_files = ['symptoms.pkl', 'diseases.pkl', 'idx_to_disease.pkl', 'scaler.pkl']
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"File not found: {file}")
                return False
                
        # Try to load each file
        with open('symptoms.pkl', 'rb') as f:
            loaded_symptoms = pickle.load(f)
            if loaded_symptoms:  # Only update if not empty
                symptoms = loaded_symptoms
                logger.info("Symptoms loaded successfully")
                
        with open('diseases.pkl', 'rb') as f:
            diseases = pickle.load(f)
            logger.info("Diseases loaded successfully")
            
        with open('idx_to_disease.pkl', 'rb') as f:
            idx_to_disease = pickle.load(f)
            logger.info("Disease index mapping loaded successfully")
            
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")

        # Load model
        if all([symptoms, diseases, idx_to_disease, scaler]):
            input_size = len(symptoms)
            hidden_size = 128
            num_classes = len(diseases)
            model = load_model(input_size, hidden_size, num_classes)
            logger.info("Model loaded successfully")
            return True
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading model data: {e}")
        logger.error(traceback.format_exc())
        
    return False

@app.route('/')
def index():
    # No need for global declaration since we're not modifying symptoms
    logger.info("Serving index page")
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    
    if model is None or symptoms is None or idx_to_disease is None or scaler is None:
        logger.error("Model not fully loaded, cannot make prediction")
        return render_template('index.html', symptoms=symptoms, 
                             error="El modelo no está disponible. Por favor, inténtelo más tarde.")

    try:
        dark_mode = request.form.get('dark_mode') == 'true'
        selected_symptoms = request.form.getlist('symptoms')
        logger.info(f"Selected symptoms: {selected_symptoms}")

        if not selected_symptoms:
            return render_template('index.html', symptoms=symptoms, 
                                 error="Por favor seleccione al menos un síntoma")

        translated_symptoms = [symptom_translation.get(symptom, symptom) for symptom in selected_symptoms]

        predicted_disease, probabilities = predict_disease(
            model, selected_symptoms, symptoms, scaler, idx_to_disease
        )
        
        logger.info(f"Prediction successful: {predicted_disease}")

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
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(traceback.format_exc())
        return render_template('index.html', symptoms=symptoms, 
                             error="Ha ocurrido un error durante la predicción.")

@app.route('/update_chart', methods=['POST'])
def update_chart():
    try:
        logger.info("Received chart update request")
        
        if model is None or symptoms is None or idx_to_disease is None or scaler is None:
            logger.error("Model not fully loaded, cannot update chart")
            return {"error": "Model not available"}, 500
            
        dark_mode = request.form.get('dark_mode') == 'true'
        selected_symptoms = ['fiebre', 'tos', 'dolor_de_cabeza']

        predicted_disease, probabilities = predict_disease(
            model, selected_symptoms, symptoms, scaler, idx_to_disease
        )

        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [(idx_to_disease[idx], float(probabilities[idx]) * 100) for idx in top_indices]

        chart_data = generate_pie_chart(top_diseases, is_dark_mode=dark_mode)
        return {"chart_data": chart_data}
    except Exception as e:
        logger.error(f"Error updating chart: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}, 500

# Try to load data at startup
model_loaded = load_data()
if not model_loaded:
    logger.warning("Model initialization failed. App will run with limited functionality.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
