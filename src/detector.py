import torch
from ultralytics import YOLO
import traceback
import sys

class PersonDetector:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        try:
            model = YOLO(model_path)
            if torch.cuda.is_available():
                print("CUDA disponível. Forçando modelo para GPU...")
                model.to('cuda')
            
            is_cuda = next(model.parameters()).is_cuda
            print(f"Modelo YOLO carregado. Usando GPU: {is_cuda}")
            return model
        except Exception as e:
            print("Erro ao carregar YOLO:")
            traceback.print_exc()
            sys.exit(1)

    def track(self, frame):
        # device=0 forces GPU 0 if available
        device = 0 if torch.cuda.is_available() else 'cpu'
        return self.model.track(frame, persist=True, verbose=False, device=device)
