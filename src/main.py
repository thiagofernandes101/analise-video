import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2 as cv
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import traceback

import torch

# --- 1. Configuração Inicial ---

# Tenta encontrar o vídeo de forma dinâmica (funciona no host e no container)
# No container, o vídeo está em /app/videos/...
# No host, está relativo ao script.
base_dir = os.path.dirname(os.path.abspath(__file__))
video_filename = 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'

# Caminhos possíveis
possible_paths = [
    os.path.join(base_dir, '../videos', video_filename), # Desenvolvimento local
    os.path.join('/app/videos', video_filename)          # Docker
]

video_path = None
for path in possible_paths:
    if os.path.exists(path):
        video_path = path
        break

if video_path is None:
    print(f"ERRO: Vídeo não encontrado em nenhum dos caminhos: {possible_paths}")
    # Fallback para webcam se o vídeo não existir (opcional)
    # video_path = 0 
    exit(1)

print(f"Usando vídeo: {video_path}")

# --- 2. Carregar o Modelo YOLOv8-Pose (GPU) ---
# "n" (nano) é o mais rápido. "m" (medium) é mais preciso, mas lento.
# Comece com 'yolov8n-pose.pt' (nano). Se sua 3060 aguentar, tente 'yolov8m-pose.pt'
try:
    model = YOLO("yolov8n-pose.pt")
    
    # Força o uso da GPU se disponível
    if torch.cuda.is_available():
        print("CUDA disponível. Forçando modelo para GPU...")
        model.to('cuda')
    
    # Verifica onde o modelo está
    # Nota: Ultralytics pode não atualizar .parameters() imediatamente, mas o .to() funciona.
    is_cuda = next(model.parameters()).is_cuda
    print(f"Modelo YOLO carregado. Usando GPU: {is_cuda}")
    
except Exception as e:
    print("Erro ao carregar YOLO:")
    traceback.print_exc()
    exit(1)

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
    # device=0 força o uso da primeira GPU
    results = model.track(frame, persist=True, verbose=False, device=0 if torch.cuda.is_available() else 'cpu')
    
    # Obter os resultados (caixas, poses, IDs)
    if results and len(results) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
        track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints.xy is not None else []
    else:
        boxes, track_ids, keypoints = [], [], []

    # --- 5. Loop sobre CADA PESSOA RASTREADA ---
    
    current_tracked_persons = [] # Guarda quem foi visto neste frame
    
    if len(boxes) > 0 and len(track_ids) > 0:
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

    # --- 9. Exibição com Tratamento de Erros ---
    try:
        if os.environ.get('DISPLAY'):
            cv.imshow('YOLOv8-Pose Tracking & Emotion Analysis (GPU)', frame)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            # Se não tiver display, apenas imprime progresso a cada X frames
            if frame_count % 30 == 0:
                print(f"Processando frame {frame_count}...")
    except Exception as e:
        print("Erro ao exibir frame (cv.imshow):")
        traceback.print_exc()
        print("Continuando processamento sem exibição...")
        # Desabilita display para evitar spam de erros
        os.environ.pop('DISPLAY', None)

video_capture.release()
cv.destroyAllWindows()