import torch
from ultralytics import YOLO
import traceback
import sys

class PersonDetector:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        try:
            # Carrega modelo YOLO-Pose
            model = YOLO(model_path)
            if torch.cuda.is_available():
                print("CUDA disponível. Forçando modelo para GPU...")
                model.to('cuda')
            
            is_cuda = next(model.parameters()).is_cuda
            print(f"Modelo YOLO Pose carregado. Usando GPU: {is_cuda}")
            return model
        except Exception as e:
            print("Erro ao carregar YOLO:")
            traceback.print_exc()
            sys.exit(1)

    def track(self, frame):
        """
        Rastreia pessoas e retorna keypoints.
        """
        # device=0 forces GPU 0 if available
        device = 0 if torch.cuda.is_available() else 'cpu'
        
        # verbose=False para limpar o console
        # persist=True para manter IDs de rastreamento
        return self.model.track(frame, persist=True, verbose=False, device=device, classes=[0]) # classe 0 = person

    def warmup(self):
        print("Aquecendo PersonDetector (YOLO)...")
        try:
            # Cria uma imagem preta pequena para inferência inicial
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.track(dummy_img)
            print("PersonDetector pronto.")
        except Exception as e:
            print(f"Erro no warmup do PersonDetector: {e}")
