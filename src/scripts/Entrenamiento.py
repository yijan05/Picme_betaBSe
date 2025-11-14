import cv2
import os
import numpy as np

# Ruta donde tienes las carpetas de cada persona
dataPath = 'C:/Users/yijan/OneDrive/Escritorio/proyecto/banco imagenes'


peopleList = os.listdir(dataPath)
print("Personas detectadas:", peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)

    # 🔹 Evitar error si hay archivos que no son carpetas
    if not os.path.isdir(personPath):
        print(f"⚠ '{nameDir}' no es una carpeta, se omite.")
        continue

    print(f"\n📂 Carpeta: {nameDir}")

    for fileName in os.listdir(personPath):
        filePath = os.path.join(personPath, fileName)
        image = cv2.imread(filePath, 0)  # Leer en escala de grises

        if image is None:
            print(f"⚠ No se pudo leer: {fileName}")
            continue

        facesData.append(image)
        labels.append(label)
        print(f"✅ Cargada: {fileName}")

    label += 1

# ================================
# ENTRENAMIENTO con LBPH
# ================================
if facesData and labels:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("\n🚀 Entrenando modelo...")
    face_recognizer.train(facesData, np.array(labels))

    # Guardar modelo
    face_recognizer.write('modeloLBPH.xml')
    print("💾 Modelo almacenado como 'modeloLBPH.xml'")
else:
    print("❌ No se encontraron imágenes válidas para entrenar.")
