# 1. Imagem Base Limpa (Python 3.10)
# Usamos uma imagem limpa para ter controle total sobre a instalação do PyTorch/CUDA.
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 2. Dependências do Sistema (para OpenCV e X11)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalação do PyTorch com CUDA (Explícita)
# Usamos o index-url oficial para CUDA 12.1 (compatível com drivers recentes).
# Isso garante que o torch venha com as bibliotecas CUDA embutidas (runtime).
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instala o YOLO
# Como o torch já está instalado, ele deve respeitar a versão existente.
RUN pip install --no-cache-dir ultralytics

# 5. TensorFlow (CPU) e Keras
# Mantemos as versões travadas e aspas para evitar erros de shell.
RUN pip install --no-cache-dir \
    "tensorflow-cpu<=2.16.1" \
    "tf-keras~=2.16.0" \
    "numpy<=1.26.4" \
    "protobuf<5" \
    "opencv-python<=4.9.0.80"

# 6. DeepFace e Mediapipe
RUN pip install --no-cache-dir --no-deps deepface mediapipe

# 7. Dependências extras do DeepFace
RUN pip install --no-cache-dir mtcnn retina-face gdown pandas tqdm Pillow

# 8. Copia o Projeto
COPY src ./src
COPY videos ./videos
COPY *.pt ./

# 9. Comando Padrão
CMD ["python", "src/main.py"]