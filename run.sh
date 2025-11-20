#!/bin/bash

# Nome da imagem
IMAGE_NAME="analise-video-gpu"

echo "--- 1. Construindo a Imagem Docker ---"
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "Erro na construção da imagem."
    exit 1
fi

echo "--- 2. Configurando Permissões do X11 (Display) ---"
# Permite que o container acesse o servidor X local
if command -v xhost &> /dev/null; then
    xhost +local:docker
else
    echo "AVISO: Comando 'xhost' não encontrado."
    echo "A visualização do vídeo pode falhar se as permissões do X11 não estiverem configuradas."
    echo "Para corrigir, instale: sudo apt-get install x11-xserver-utils"
fi

echo "--- 3. Executando o Container com GPU e Display ---"
# --gpus all: Habilita acesso à GPU
# -e DISPLAY=$DISPLAY: Passa a variável de ambiente DISPLAY
# -v /tmp/.X11-unix:/tmp/.X11-unix: Monta o socket do X11
# --rm: Remove o container ao sair
docker run --rm -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.deepface:/root/.deepface \
    $IMAGE_NAME

# Opcional: Verificar se o container viu a GPU
# docker run --rm --gpus all $IMAGE_NAME nvidia-smi

echo "--- 4. Diagnóstico de GPU ---"
echo "Verificando Runtime NVIDIA no Docker (Host):"
docker info | grep -i runtime
if [ $? -ne 0 ]; then
    echo "ALERTA: Runtime 'nvidia' não encontrado no Docker info."
    echo "Verifique se o nvidia-container-toolkit está configurado em /etc/docker/daemon.json"
fi

echo "Verificando PyTorch CUDA dentro do Container:"
docker run --rm --gpus all $IMAGE_NAME python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

