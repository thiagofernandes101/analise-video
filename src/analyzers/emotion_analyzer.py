import threading
import queue
import time
from deepface import DeepFace
from config import Config
from models.emotion_result import EmotionResult

class EmotionAnalyzer:
    def __init__(self, config: Config = Config()):
        # Queue holds (track_id, face_img, frame_number) tuples
        queue_size = config.emotion.QUEUE_MAX_SIZE
        self.emotion_queue = queue.Queue(maxsize=queue_size)
        self.emotion_cache = {}  # {track_id: EmotionResult}
        self.running = True
        self._config = config
        self._start_worker()

    def _start_worker(self):
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        worker_thread.start()
        print("Worker de emoção iniciado.")

    def stop(self):
        self.running = False

    def _worker(self):
        while self.running:
            try:
                # Timeout curto para verificar self.running
                task = self.emotion_queue.get(timeout=0.1)
                track_id, face_img, frame_number = task
                
                try:
                    # Analisa apenas a emoção da imagem recortada (face_img)
                    result = DeepFace.analyze(
                        img_path=face_img, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='skip'  # Pular detecção pois já é um crop
                    )
                    
                    dominant_emotion = 'neutral'
                    confidence = 0.0
                    
                    if isinstance(result, list) and len(result) > 0:
                        analysis = result[0]
                        dom_emo = analysis['dominant_emotion']
                        confidence = analysis['emotion'][dom_emo]
                        
                        threshold = self._config.emotion.CONFIDENCE_THRESHOLD
                        if confidence >= threshold:
                            dominant_emotion = dom_emo
                    
                    # Atualiza cache com EmotionResult
                    emotion_result = EmotionResult(
                        emotion=dominant_emotion,
                        confidence=confidence,
                        frame_number=frame_number
                    )
                    self.emotion_cache[track_id] = emotion_result
                    
                except Exception as e:
                    # Se falhar, mantém o anterior ou define Unknown
                    print(f"⚠️ ERRO ao analisar emoção para track_id={track_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    if track_id not in self.emotion_cache:
                         self.emotion_cache[track_id] = EmotionResult.unknown(frame_number)
                finally:
                    self.emotion_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro fatal no worker de emoção: {e}")
                time.sleep(1)

    def analyze_async(self, track_id, face_img, current_frame):
        """Enfileira tarefa de análise se a fila não estiver cheia."""
        if not self.emotion_queue.full():
            self.emotion_queue.put((track_id, face_img, current_frame))
            
            # Indica que está processando se não tiver valor anterior
            if track_id not in self.emotion_cache:
                self.emotion_cache[track_id] = EmotionResult.analyzing(current_frame)

    def get_emotion(self, track_id) -> EmotionResult:
        """Retorna EmotionResult para um track_id."""
        return self.emotion_cache.get(track_id, EmotionResult.unknown(0))

    def clean_cache(self, current_tracked_ids):
        """Limpa IDs que não estão mais sendo rastreados."""
        all_known_ids = list(self.emotion_cache.keys())
        for track_id in all_known_ids:
            if track_id not in current_tracked_ids:
                del self.emotion_cache[track_id]
