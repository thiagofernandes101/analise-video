import threading
import queue
from deepface import DeepFace
from config import Config

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_queue = queue.Queue(maxsize=5)
        self.emotion_cache = {}
        self._start_worker()

    def _start_worker(self):
        worker_thread = threading.Thread(target=self._worker, daemon=True)
        worker_thread.start()
        print("Worker de emoção iniciado.")

    def _worker(self):
        while True:
            try:
                task = self.emotion_queue.get(timeout=1)
                track_id, face_img, current_frame = task
                
                try:
                    result = DeepFace.analyze(
                        face_img, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='retinaface'
                    )
                    
                    dominant_emotion = 'neutral'
                    if isinstance(result, list) and len(result) > 0:
                        confidence = result[0]['emotion'][result[0]['dominant_emotion']]
                        if confidence >= Config.EMOTION_CONFIDENCE_THRESHOLD:
                            dominant_emotion = result[0]['dominant_emotion']
                    
                    self.emotion_cache[track_id] = (dominant_emotion, current_frame)
                    
                except Exception:
                    if track_id not in self.emotion_cache:
                         self.emotion_cache[track_id] = ('N/A', current_frame)
                    else:
                         old_emotion = self.emotion_cache[track_id][0]
                         self.emotion_cache[track_id] = (old_emotion, current_frame)

                finally:
                    self.emotion_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro fatal no worker: {e}")
                break

    def analyze_async(self, track_id, face_img, current_frame):
        if not self.emotion_queue.full():
            self.emotion_queue.put((track_id, face_img, current_frame))
            
            # Update cache provisionally
            current_emotion = self.emotion_cache.get(track_id, ("Analyzing...", 0))[0]
            self.emotion_cache[track_id] = (current_emotion, current_frame)

    def get_emotion(self, track_id):
        return self.emotion_cache.get(track_id, ("...", 0))

    def clean_cache(self, current_tracked_ids):
        all_known_ids = list(self.emotion_cache.keys())
        for track_id in all_known_ids:
            if track_id not in current_tracked_ids:
                del self.emotion_cache[track_id]
