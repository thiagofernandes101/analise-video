import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2 as cv
import numpy as np
from deepface import DeepFace
import mediapipe as mp  # <-- 1. Importar MediaPipe

script_directory = os.path.dirname(__file__)
video_path = '/home/thiagofernandes101/projects/fiap/analise-video/videos/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
proto_path = os.path.join(script_directory, "deploy.prototxt")
model_path = os.path.join(script_directory, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv.dnn.readNetFromCaffe(proto_path, model_path)

# --- 2. Configuração do MediaPipe Pose ---
mp_pose = mp.solutions.pose
# Você pode ajustar os parâmetros aqui (ex: min_detection_confidence)
pose = mp_pose.Pose() 
mp_drawing = mp.solutions.drawing_utils
# -------------------------------------

video_capture = cv.VideoCapture(video_path)

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("End of video stream or cannot read the video.")
        break
    
    # O MediaPipe e o DeepFace precisam de RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # --- 3. Processamento de Pose (MediaPipe) ---
    # Processa o frame INTEIRO para encontrar poses
    pose_results = pose.process(rgb_frame)
    # ------------------------------------------
    
    (h, w) = frame.shape[:2]
    
    # --- Seu código existente de Detecção Facial (DNN) ---
    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300, 300)), 
        1.0,
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    net.setInput(blob)
    detections = net.forward()
    
    # Loop sobre as detecções de ROSTO
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            x1 = max(0, startX)
            y1 = max(0, startY)
            x2 = min(w, endX)
            y2 = min(h, endY)

            if x2 <= x1 or y2 <= y1:
                continue

            # --- Análise de Emoção (DeepFace) ---
            # O ROI já está em RGB (do 'rgb_frame')
            region_of_interest = rgb_frame[y1:y2, x1:x2]

            try:
                # O enforce_detection=False é crucial aqui
                result = DeepFace.analyze(region_of_interest, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    dominant_emotion = result[0].get('dominant_emotion', 'N/A')
                else:
                    dominant_emotion = result.get('dominant_emotion', 'N/A')
            except Exception:
                dominant_emotion = 'N/A'

            # Desenha o retângulo da FACE (em verde)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = y1 - 10 if (y1 - 10) > 0 else (y2 + 20)
            cv.putText(frame, dominant_emotion, (x1, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # --- 4. Desenhar os resultados da POSE (MediaPipe) ---
    # Desenha o esqueleto da pose sobre o frame original (BGR)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            # Estilizando os pontos e conexões
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
    # --------------------------------------------------
        
    cv.imshow('Face, Emotion, and Pose Analysis', frame) # Título da janela atualizado
    
    if cv.waitKey(1) == ord('q'):
        break

video_capture.release()
pose.close() # <-- 5. Liberar o recurso de pose
cv.destroyAllWindows()