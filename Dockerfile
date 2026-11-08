# Trabajador GPU de Series IA (RunPod Serverless + ComfyUI).
#
# Base: runpod/worker-comfyui 5.10.0-base (verificado en Docker Hub el 7/11/2026;
# la release 5.11.0 de GitHub NO tiene imagen publicada todavía).
#   - Ubuntu 24.04, CUDA 12.8.1 (nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04)
#   - Python 3.12 en /opt/venv, torch 2.11.0+cu128, comfy-cli 1.13.0
#   - ComfyUI 0.34.0 en /comfyui, runpod + requests + websocket-client ya instalados
#   - /start.sh arranca ComfyUI y luego `python -u /handler.py`
#   - trae /usr/local/bin/comfy-node-install (para nodos del registro de Comfy)
# Fuente: https://github.com/runpod-workers/worker-comfyui/blob/5.10.0/Dockerfile
#
# ComfyUI 0.34.0 no alcanza: FastH3 pide 0.36.0+ (https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
# y LTX-2.5 salió después. Por eso se cambia ComfyUI a la versión fija v0.44.2.
FROM runpod/worker-comfyui:5.10.0-base

ARG COMFYUI_VERSION=0.44.2
# Commits fijos de los nodos personalizados (7/11/2026). Los 4 flujos actuales
# usan solo nodos nativos; estos dos quedan para variantes (VHS_VideoCombine,
# utilidades de KJ) y el handler ya entiende la salida "gifs" de VHS.
ARG VHS_COMMIT=4d907bee61e92c2e65af3bd6383a4e4d356126d1
ARG KJNODES_COMMIT=ab72eda7ffad6280422d71ab7d54d3fb87c0ba18

ENV PIP_NO_INPUT=1 \
    PYTHONUNBUFFERED=1

# 1) ComfyUI v0.44.2 encima del de la base (se conservan custom_nodes y models).
RUN set -eux; \
    cd /tmp; \
    wget -q "https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v${COMFYUI_VERSION}.tar.gz" -O comfyui.tgz; \
    tar xzf comfyui.tgz; \
    cp -a "ComfyUI-${COMFYUI_VERSION}/." /comfyui/; \
    rm -rf comfyui.tgz "ComfyUI-${COMFYUI_VERSION}"; \
    uv pip install --python /opt/venv/bin/python -r /comfyui/requirements.txt "transformers>=4.50.3,<5" "huggingface-hub<1.0"; \
    python -c "import sys; sys.path.insert(0, '/comfyui'); import comfyui_version as v; assert v.__version__ == '${COMFYUI_VERSION}', v.__version__; print('ComfyUI', v.__version__)"

# 2) Nodos personalizados con commit fijo.
RUN set -eux; \
    cd /comfyui/custom_nodes; \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git; \
    git -C ComfyUI-VideoHelperSuite checkout "${VHS_COMMIT}"; \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git; \
    git -C ComfyUI-KJNodes checkout "${KJNODES_COMMIT}"; \
    for r in ComfyUI-VideoHelperSuite/requirements.txt ComfyUI-KJNodes/requirements.txt; do \
      [ -f "$r" ] && uv pip install --python /opt/venv/bin/python -r "$r"; \
    done; \
    uv pip install --python /opt/venv/bin/python "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# 3) Rutas del volumen de red (agrega diffusion_models, text_encoders y latent_upscale_models).
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# 4) Prueba de humo en CPU (igual que la base): si un nodo rompe el arranque, falla aquí y no en RunPod.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# 5) Nuestro handler (reemplaza al de la base; /start.sh lo ejecuta como /handler.py) y los flujos.
COPY handler.py /handler.py
COPY flujos/ /flujos/

# Tiempo máximo por toma dentro del handler (s). El «Execution timeout» del endpoint debe ser mayor.
ENV TIEMPO_MAX_SEG=1200 \
    FLUJOS_DIR=/flujos

CMD ["/start.sh"]
