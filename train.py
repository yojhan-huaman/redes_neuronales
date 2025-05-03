import pickle
from model import prepare_data, DiseaseClassifier, train_model, evaluate_model, save_model

def main():
    print("Iniciando entrenamiento del clasificador de enfermedades...")
    
    dataset_path = 'dataset.csv'
    print(f"Cargando dataset desde {dataset_path}...")
    
    X_train, X_test, y_train, y_test, symptoms, diseases, disease_to_idx, idx_to_disease, scaler = prepare_data(dataset_path)
    
    print(f"Dataset cargado. Contiene {len(symptoms)} síntomas y {len(diseases)} enfermedades.")
    
    input_size = len(symptoms)
    hidden_size = 128
    num_classes = len(diseases)
    
    model = DiseaseClassifier(input_size, hidden_size, num_classes)
    print("Modelo creado. Estructura:")
    print(model)
    
    print("\nIniciando entrenamiento...")
    train_model(model, X_train, y_train, epochs=100, learning_rate=0.001)
    
    print("\nEvaluando modelo...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    save_model(model, 'disease_classifier.pth')
    
    print("Guardando datos adicionales...")
    with open('symptoms.pkl', 'wb') as f:
        pickle.dump(symptoms, f)
    
    with open('diseases.pkl', 'wb') as f:
        pickle.dump(diseases, f)
    
    with open('idx_to_disease.pkl', 'wb') as f:
        pickle.dump(idx_to_disease, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Entrenamiento completado. Resumen:")
    print(f"- Número de síntomas: {input_size}")
    print(f"- Número de enfermedades: {num_classes}")
    print(f"- Precisión del modelo: {accuracy:.4f}")
    print("- Archivos guardados: disease_classifier.pth, symptoms.pkl, diseases.pkl, idx_to_disease.pkl, scaler.pkl")
    print("\nAhora puede ejecutar la aplicación web con 'python app.py'")

if __name__ == "__main__":
    main()
