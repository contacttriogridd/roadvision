# RoadVision inference image — FastAPI (REST + web UI) with Gradio mounted at /ui.
# Works locally (docker run -p 7860:7860) and as a Hugging Face Docker Space (port 7860).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/mpl HF_HOME=/tmp/hf

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU wheels keep the image ~1.5 GB instead of ~8 GB with CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY --chown=user configs ./configs
COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user app ./app
# trained weights + calibration are baked in (no download needed at start-up)
COPY --chown=user models ./models

USER user
ENV RV_CONFIG=/home/user/app/configs/default.yaml RV_OVERRIDE=/home/user/app/configs/fast45.yaml PYTHONPATH=/home/user/app/src:/home/user/app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:7860/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
