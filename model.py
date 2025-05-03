import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def prepare_data(dataset_path):
    # Cargar y preprocesar el dataset
    data = pd.read_csv(dataset_path)
    
    # Separar características y etiquetas
    X = data.iloc[:, :-1].values  # Todos los síntomas
    y = data.iloc[:, -1].values   # Enfermedades
    
    # Crear una lista de síntomas y enfermedades
    symptoms = list(data.columns[:-1])
    diseases = list(set(y))
    
    # Mapear enfermedades a índices
    disease_to_idx = {disease: idx for idx, disease in enumerate(diseases)}
    idx_to_disease = {idx: disease for disease, idx in disease_to_idx.items()}
    
    # Convertir etiquetas a índices
    y_idx = np.array([disease_to_idx[disease] for disease in y])
    
    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_idx, test_size=0.2, random_state=42
    )
    
    # Convertir a tensores de PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return (
        X_train, X_test, y_train, y_test, 
        symptoms, diseases, disease_to_idx, idx_to_disease, 
        scaler
    )

def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entrenamiento
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward y optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    print('Entrenamiento completado')

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Precisión del modelo: {accuracy:.4f}')
    return accuracy

def save_model(model, path='disease_classifier.pth'):
    torch.save(model.state_dict(), path)
    print(f'Modelo guardado en {path}')

def load_model(input_size, hidden_size, num_classes, path='disease_classifier.pth'):
    model = DiseaseClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_disease(model, symptoms_input, all_symptoms, scaler, idx_to_disease):
    # Crear un vector de características con ceros
    symptoms_vector = np.zeros(len(all_symptoms))
    
    # Establecer en 1 los síntomas presentes
    for symptom in symptoms_input:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            symptoms_vector[idx] = 1
    
    # Normalizar usando el mismo escalador
    symptoms_vector = scaler.transform([symptoms_vector])
    
    # Convertir a tensor de PyTorch
    symptoms_tensor = torch.FloatTensor(symptoms_vector)
    
    # Predicción
    with torch.no_grad():
        outputs = model(symptoms_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_disease = idx_to_disease[predicted_idx.item()]
    
    # Devolver la enfermedad predicha y las probabilidades
    return predicted_disease, probabilities.numpy()[0]

if __name__ == "__main__":
    # Este código se ejecutará si ejecutamos este archivo directamente
    # para entrenar y guardar el modelo
    
    dataset_path = 'dataset.csv'
    
    # Preparar datos
    X_train, X_test, y_train, y_test, symptoms, diseases, disease_to_idx, idx_to_disease, scaler = prepare_data(dataset_path)
    
    # Crear y entrenar modelo
    input_size = len(symptoms)
    hidden_size = 128
    num_classes = len(diseases)
    
    model = DiseaseClassifier(input_size, hidden_size, num_classes)
    train_model(model, X_train, y_train)
    
    # Evaluar modelo
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Guardar modelo
    save_model(model)
    
    print(f"Número de síntomas: {input_size}")
    print(f"Número de enfermedades: {num_classes}")