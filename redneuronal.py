import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Definir las rutas de las carpetas
base_dir = "imagenes"
classes = ["semaforo", "caja"]

# Función para cargar imágenes y etiquetas
def cargar_imagenes_y_etiquetas(base_dir, classes):
    data = []
    labels = []
    for label, class_name in enumerate(classes):
        folder_path = os.path.join(base_dir, class_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                with Image.open(file_path) as img:
                    img = img.resize((64, 64))  # Redimensionar a 64x64 píxeles
                    img_array = np.array(img.convert("L"))  # Convertir a escala de grises
                    data.append(img_array.flatten())  # Aplanar la imagen
                    labels.append(label)
            except Exception as e:
                print(f"Error al cargar {file_path}: {e}")
    return np.array(data), np.array(labels)

# Cargar datos y etiquetas
data, labels = cargar_imagenes_y_etiquetas(base_dir, classes)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes))


import joblib

# Guardar el modelo en un archivo
joblib.dump(model, "modelo_red_neuronal.pkl")
print("Modelo guardado como 'modelo_red_neuronal.pkl'")


