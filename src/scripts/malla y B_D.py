import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================================
# RUTA Y MODELO ENTRENADO
# ================================
dataPath = 'C:/Users/yijan/OneDrive/Escritorio/proyecto/banco imagenes'
imagePaths = os.listdir(dataPath)
print("Personas detectadas:", imagePaths)

# Cargar modelo LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPH.xml')

# Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================================
# MEDIAPIPE MALLA FACIAL
# ================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ================================
# VARIABLES DE PARPADEO
# ================================
EAR_THRESH = 0.26
NUM_FRAMES = 2
blink_counter = 0
aux_counter = 0
mesh_active = False
last_z = None

# Índices de ojos
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

# ================================
# FUNCIONES
# ================================
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

# ================================
# CAPTURA DE VIDEO
# ================================
cap = cv2.VideoCapture("http://192.168.206.30:8080/video")
cv2.namedWindow("Reconocimiento + Parpadeo + Malla", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconocimiento + Parpadeo + Malla", 640, 480)



with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ==========================
        # DETECCIÓN DE ROSTROS
        # ==========================
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro_resized = cv2.resize(rostro, (200,200), interpolation=cv2.INTER_CUBIC)

            # ==========================
            # RECONOCIMIENTO FACIAL
            # ==========================
            label, confidence = face_recognizer.predict(rostro_resized)

            if confidence < 85:
                person_name = imagePaths[label]
                color = (0,255,0)
            else:
                person_name = "Desconocido"
                color = (0,0,255)

            # Dibujar rectángulo y nombre (solo si luego cumple el parpadeo)
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x,y-10), 1, 1.3, color, 2, cv2.LINE_AA)

            # ==========================
            # PROCESAR MALLA FACIAL
            # ==========================
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    coordinates_left_eye = []
                    coordinates_right_eye = []
                    z_movement = False

                    for idx in index_left_eye:
                        x_eye = int(face_landmarks.landmark[idx].x * frame.shape[1])
                        y_eye = int(face_landmarks.landmark[idx].y * frame.shape[0])
                        z_eye = face_landmarks.landmark[idx].z
                        coordinates_left_eye.append([x_eye, y_eye])
                        if last_z is not None and abs(last_z - z_eye) > 0.005:
                            z_movement = True

                    for idx in index_right_eye:
                        x_eye = int(face_landmarks.landmark[idx].x * frame.shape[1])
                        y_eye = int(face_landmarks.landmark[idx].y * frame.shape[0])
                        z_eye = face_landmarks.landmark[idx].z
                        coordinates_right_eye.append([x_eye, y_eye])
                        if last_z is not None and abs(last_z - z_eye) > 0.005:
                            z_movement = True

                    last_z = z_eye

                    # Calcular EAR
                    ear_left = eye_aspect_ratio(coordinates_left_eye)
                    ear_right = eye_aspect_ratio(coordinates_right_eye)
                    ear = (ear_left + ear_right) / 2

                    # Contar parpadeos
                    if ear < EAR_THRESH:
                        aux_counter += 1
                    else:
                        if aux_counter >= NUM_FRAMES:
                            blink_counter += 1
                        aux_counter = 0

                    # Activar malla SOLO si parpadeó 10 veces
                    if blink_counter >= 10 and z_movement:
                        mesh_active = True

                    # Dibujar malla si está activa y la persona es reconocida
                    if mesh_active and person_name != "Desconocido":
                        mp_drawing.draw_landmarks(
                            frame, face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )

        # Mostrar contador de parpadeos
        cv2.putText(frame, f"Parpadeos: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento + Parpadeo + Malla", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
