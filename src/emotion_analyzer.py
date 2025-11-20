import threading
import queue
import time
from deepface import DeepFace
from config import Config

class EmotionAnalyzer:
    def __init__(self):
        # Queue agora guarda apenas (track_id, face_img, timestamp)
        # maxsize pequeno para evitar acumulo de frames antigos
        self.emotion_queue = queue.Queue(maxsize=2)
        self.emotion_cache = {} # {track_id: (emotion, timestamp)}
        self.running = True
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
                track_id, face_img, timestamp = task
                
                try:
                    # Analisa apenas a emoção da imagem recortada (face_img)
                    # actions=['emotion']
                    # enforce_detection=False pois já recortamos o rosto
                    result = DeepFace.analyze(
                        img_path=face_img, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='skip' # Importante: pular detecção pois já é um crop
                    )
                    
                    dominant_emotion = 'neutral'
                    if isinstance(result, list) and len(result) > 0:
                        # DeepFace retorna lista de dicts
                        analysis = result[0]
                        dom_emo = analysis['dominant_emotion']
                        confidence = analysis['emotion'][dom_emo]
                        
                        if confidence >= Config.EMOTION_CONFIDENCE_THRESHOLD:
                            dominant_emotion = dom_emo
                    
                    # Atualiza cache
                    self.emotion_cache[track_id] = (dominant_emotion, timestamp)
                    
                except Exception as e:
                    # Se falhar, mantem o anterior ou define N/A
                    # print(f"Erro na analise de emoção: {e}")
                    if track_id not in self.emotion_cache:
                         self.emotion_cache[track_id] = ('N/A', timestamp)
                finally:
                    self.emotion_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro fatal no worker de emoção: {e}")
                time.sleep(1) # Evita loop rápido em caso de erro persistente

    def analyze_async(self, track_id, face_img, current_frame):
        """
        Enfileira tarefa de análise. Se a fila estiver cheia,
        descarta a tarefa mais antiga (se possível) ou ignora a nova 
        para priorizar o tempo real (drop frame strategy).
        Neste caso, como queremos o mais recente, se estiver cheia, 
        podemos tentar esvaziar um item e colocar o novo, ou apenas ignorar.
        Para simplicidade e evitar lag: se cheia, ignora este frame.
        """
        if not self.emotion_queue.full():
            self.emotion_queue.put((track_id, face_img, current_frame))
            
            # Atualiza cache provisoriamente para indicar que está processando
            # apenas se não tiver valor anterior recente
            if track_id not in self.emotion_cache:
                self.emotion_cache[track_id] = ("Analyzing...", current_frame)
        else:
            # Opcional: Poderíamos remover o item mais antigo da fila para colocar o novo
            # mas queue.Queue não suporta isso facilmente. 
            # Com maxsize=2, se estiver cheia, o worker está ocupado. Melhor pular.
            pass

    def get_emotion(self, track_id):
        return self.emotion_cache.get(track_id, ("...", 0))

    def clean_cache(self, current_tracked_ids):
        # Limpa IDs que não estão mais sendo rastreados para economizar memória
        # Mas dá uma margem de tempo/frames antes de limpar para evitar flicker
        all_known_ids = list(self.emotion_cache.keys())
        for track_id in all_known_ids:
            if track_id not in current_tracked_ids:
                del self.emotion_cache[track_id]
