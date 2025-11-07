import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2 as cv
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO # <-- 1. Importar YOLO

# --- 1. Configuração Inicial ---

video_path = '/home/thiagofernandes101/projects/fiap/analise-video/videos/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
# video_path = 0 # Para webcam

# --- 2. Carregar o Modelo YOLOv8-Pose (GPU) ---
# "n" (nano) é o mais rápido. "m" (medium) é mais preciso, mas lento.
# Comece com 'yolov8n-pose.pt' (nano). Se sua 3060 aguentar, tente 'yolov8m-pose.pt'
model = YOLO("yolov8n-pose.pt")
print("Modelo YOLO carregado. Usando GPU:", next(model.parameters()).is_cuda)

video_capture = cv.VideoCapture(video_path)

# --- 3. Variáveis para Rastreamento de Emoção ---
frame_count = 0
EMOTION_ANALYSIS_INTERVAL = 15  # Analisa emoção a cada 15 frames POR PESSOA
EMOTION_CONFIDENCE_THRESHOLD = 40.0 # % de confiança para aceitar uma emoção

# Cache para guardar a emoção de cada pessoa rastreada
# Formato: { track_id: (emotion, last_analysis_frame) }
emotion_cache = {}

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("End of video stream.")
        break
        
    frame_count += 1
    (h, w) = frame.shape[:2]
    
    # O DeepFace precisa de RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # --- 4. Rodar YOLOv8-Pose com Tracking (GPU) ---
    # .track() faz detecção, pose E rastreamento de uma só vez
    # 'persist=True' diz ao tracker para lembrar dos IDs entre frames
    results = model.track(frame, persist=True, verbose=False)
    
    # Obter os resultados (caixas, poses, IDs)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
    keypoints = results[0].keypoints.xy.cpu().numpy()

    # --- 5. Loop sobre CADA PESSOA RASTREADA ---
    
    current_tracked_persons = [] # Guarda quem foi visto neste frame
    
    for (box, track_id, kps) in zip(boxes, track_ids, keypoints):
        
        current_tracked_persons.append(track_id)
        
        (x1, y1, x2, y2) = box.astype("int")
        (x1, y1) = (max(0, x1), max(0, y1))
        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))
            
        dominant_emotion = "..." # Placeholder
        
        # --- 6. Lógica de Análise de Emoção (Cache por ID) ---
        
        # Este frame é um "frame de análise" PARA ESTA PESSOA?
        is_analysis_frame = False
        if track_id not in emotion_cache:
            is_analysis_frame = True # Primeira vez vendo essa pessoa
        else:
            last_frame_seen = emotion_cache[track_id][1]
            if frame_count - last_frame_seen >= EMOTION_ANALYSIS_INTERVAL:
                is_analysis_frame = True

        if is_analysis_frame:
            # --- RODA A ANÁLISE PESADA ---
            if x2 > x1 and y2 > y1:
                region_of_interest = rgb_frame[y1:y2, x1:x2]
                try:
                    # Usamos o 'retinaface' pois é um bom backend de GPU
                    result = DeepFace.analyze(
                        region_of_interest, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='retinaface' # Tenta usar um backend mais robusto
                    )
                    
                    if isinstance(result, list) and len(result) > 0:
                        # Filtro de confiança (Resolve o problema de "sad" em rostos neutros)
                        confidence = result[0]['emotion'][result[0]['dominant_emotion']]
                        if confidence >= EMOTION_CONFIDENCE_THRESHOLD:
                            dominant_emotion = result[0]['dominant_emotion']
                        else:
                            dominant_emotion = 'neutral' # Força 'neutral' se a confiança for baixa
                            
                        # Atualiza o cache
                        emotion_cache[track_id] = (dominant_emotion, frame_count)
                        
                except Exception as e:
                    # Se falhar (ex: rosto não visível no BBox), mantém a emoção antiga
                    if track_id in emotion_cache:
                        dominant_emotion = emotion_cache[track_id][0]
                    else:
                        dominant_emotion = 'N/A'
                    emotion_cache[track_id] = (dominant_emotion, frame_count) # Atualiza mesmo na falha
        
        # Se não for um frame de análise, apenas pega o valor do cache
        if track_id in emotion_cache:
            dominant_emotion = emotion_cache[track_id][0]
        else:
            dominant_emotion = "Loading..." # Vendo pela primeira vez

        # --- 7. Desenhar os Resultados ---
        
        # Desenha BBox da Pessoa (Verde) e o Track ID
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID: {track_id} - {dominant_emotion}"
        label_y = y1 - 10 if (y1 - 10) > 0 else (y2 + 20)
        cv.putText(frame, label, (x1, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Desenha os Pontos da Pose (YOLOv8)
        for (px, py) in kps:
             if px > 0 and py > 0: # Só desenha pontos válidos
                cv.circle(frame, (int(px), int(py)), 3, (0, 0, 255), -1) # Pontos em Vermelho
    
    # --- 8. Limpeza do Cache ---
    # Remove IDs de pessoas que saíram de cena
    all_known_ids = list(emotion_cache.keys())
    for track_id in all_known_ids:
        if track_id not in current_tracked_persons:
             # Opcional: manter por alguns segundos / ou remover direto
             del emotion_cache[track_id]

    cv.imshow('YOLOv8-Pose Tracking & Emotion Analysis (GPU)', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()