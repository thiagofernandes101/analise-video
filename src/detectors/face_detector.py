import cv2
from deepface import DeepFace
import numpy as np
import traceback

class FaceDetector:
    def __init__(self, backend='yolov8m'):
        self.backend = backend
        print(f"FaceDetector iniciado com backend: {self.backend}")

    def detect_faces(self, frame):
        """
        Detecta rostos no frame.
        Retorna uma lista de tuplas (x, y, w, h) ou (x1, y1, x2, y2) dependendo da necessidade.
        Aqui retornaremos (x, y, w, h) para consistência, ou podemos converter para (x1, y1, x2, y2).
        DeepFace.extract_faces retorna 'facial_area': {'x': int, 'y': int, 'w': int, 'h': int}
        """
        try:
            # DeepFace.extract_faces pode falhar se não achar rosto, então usamos enforce_detection=False
            # Mas queremos apenas as coordenadas, não o alinhamento completo agora.
            # No entanto, extract_faces é o método padrão para pegar a região.
            
            results = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.backend,
                enforce_detection=False,
                align=False
            )
            
            faces = []
            for result in results:
                # result é um dict com 'face', 'facial_area', 'confidence'
                area = result['facial_area']
                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                confidence = result.get('confidence', 0.0)
                
                # Filtrar por confiança se necessário (DeepFace já filtra um pouco)
                if w > 0 and h > 0:
                    faces.append((x, y, w, h))
            
            return faces

        except Exception as e:
            # Em caso de erro (ex: backend não instalado ou erro interno), logar e retornar vazio
            print(f"⚠️ ERRO na detecção facial: {e}")
            traceback.print_exc()
            return []

    def warmup(self):
        print("Aquecendo FaceDetector (DeepFace)...")
        try:
            # Cria uma imagem preta pequena para inferência inicial
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.detect_faces(dummy_img)
            print("FaceDetector pronto.")
        except Exception as e:
            print(f"⚠️ Erro no warmup do FaceDetector: {e}")
            traceback.print_exc()
