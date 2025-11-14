import cv2
import os
import numpy as np

# Ruta a las imágenes
dataPath =  'C:/Users/yijan/OneDrive/Escritorio/proyecto/banco_imagenes'
imagePaths = os.listdir(dataPath)
print('Clases detectadas:', imagePaths)

# Crear el reconocedor LBPH y cargar el modelo entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Asegúrate de que el modelo entrenado exista en la misma carpeta
modelo_path = 'C:/Users/yijan/OneDrive/Escritorio/proyecto/Imagenes a reconocer/modeloLBPH.xml'
if not os.path.exists(modelo_path):
    print(f"❌ No se encontró el modelo en {modelo_path}")
    exit()

face_recognizer.read(modelo_path)

# Cargar Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Métricas
true_positives = 0
false_positives = 0
total = 0

# Abrir cámara
cap = cv2.VideoCapture(0)  # prueba con 1 si no funciona
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]

        # Predecir con el modelo
        label, confidence = face_recognizer.predict(rostro)

        if confidence < 85:  # Umbral de confianza
            person_name = imagePaths[label]
            color = (0, 255, 0)
        else:
            person_name = "Desconocido"
            color = (0, 0, 255)

        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{person_name} ({int(confidence)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 🔹 Aquí contabilizamos para calcular error
        total += 1
        if person_name != "Desconocido":
            true_positives += 1
        else:
            false_positives += 1

    cv2.imshow("Evaluación del modelo", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =============================
# 📊 Resultados de evaluación
# =============================
if total > 0:
    accuracy = (true_positives / total) * 100
    error_rate = (false_positives / total) * 100
    print(f"✅ Total evaluado: {total}")
    print(f"✔ Verdaderos positivos: {true_positives}")
    print(f"❌ Falsos positivos: {false_positives}")
    print(f"🎯 Precisión: {accuracy:.2f}%")
    print(f"⚠ Tasa de error: {error_rate:.2f}%")
else:
    print("⚠ No se detectaron rostros para evaluar.")
