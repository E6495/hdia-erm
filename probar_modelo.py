import joblib
import cv2
import numpy as np

# Cargar el modelo de la red neuronal
model_path = 'modelo_red_neuronal.pkl'  # Ruta al modelo guardado
model = joblib.load(model_path)
classes = ['semaforo', 'caja']

# Cargar una imagen de prueba
image_path = '/home/eli/imagenes/caja/1.jpg'  # Cambia esto por la ruta de tu imagen
image = cv2.imread(image_path)

# Preprocesar la imagen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
resized = cv2.resize(gray, (64, 64))  # Redimensionar a 64x64 píxeles
flattened = resized.flatten().reshape(1, -1)  # Aplanar la imagen

# Hacer la predicción
prediction = model.predict(flattened)[0]
object_detected = classes[prediction]

# Mostrar el resultado
print(f'Objeto detectado: {object_detected}')

# Guardar la imagen procesada
processed_image_path = 'procesada_imagen2.jpg'
cv2.imwrite(processed_image_path, resized)

print(f'Imagen procesada guardada en: {processed_image_path}')

