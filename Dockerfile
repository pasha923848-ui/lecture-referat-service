FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model size/device/compute type baked into the image must match what the
# container runs with at runtime (see docker-compose.yml, which keeps the
# build arg and the runtime env var in sync) — otherwise faster-whisper
# would try to fetch a different model at startup instead of using the one
# cached below.
ARG WHISPER_MODEL_SIZE=base
ARG WHISPER_DEVICE=cpu
ARG WHISPER_COMPUTE_TYPE=int8

# Download and cache the Whisper model weights at build time, so the
# container never needs network access to fetch them at runtime.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL_SIZE}', device='${WHISPER_DEVICE}', compute_type='${WHISPER_COMPUTE_TYPE}')"

# Same idea for the реферат-writing LLM: bake the GGUF weights into this
# layer at build time so generating a реферат needs no network either.
# Downloaded to a fixed path (not left in the huggingface_hub cache) and
# loaded at runtime via that literal path (see app/reference/llm.py) —
# llama_cpp.Llama.from_pretrained() always calls the HF Hub API to resolve
# its `filename` glob pattern, even for an already-cached file, and that
# call hard-fails once HF_HUB_OFFLINE=1 is set below (huggingface_hub raises
# OfflineModeIsEnabled instead of falling back to the local cache), which
# would otherwise make every реферат-generation request fail at startup.
ARG LLM_MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF
ARG LLM_MODEL_FILE=qwen2.5-3b-instruct-q4_k_m.gguf
ENV LLM_MODEL_PATH=/app/models/llm.gguf
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import shutil, os; \
src = hf_hub_download(repo_id='${LLM_MODEL_REPO}', filename='${LLM_MODEL_FILE}'); \
os.makedirs('/app/models', exist_ok=True); \
shutil.copy(src, '${LLM_MODEL_PATH}')"

# Source is copied only after the multi-GB model layers, so a code change
# rebuilds in seconds instead of re-downloading and re-exporting the models.
COPY . .

ENV STORAGE_DIR=/data
# Organized lesson folders and the ingestion-tracking DB live under the
# mounted /data volume too, so they survive container restarts/rebuilds.
ENV MATERIALS_DIR=/data/materials
ENV DB_PATH=/data/materials.db
ENV WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE}
ENV WHISPER_DEVICE=${WHISPER_DEVICE}
ENV WHISPER_COMPUTE_TYPE=${WHISPER_COMPUTE_TYPE}
ENV LLM_MODEL_REPO=${LLM_MODEL_REPO}
ENV LLM_MODEL_FILE=${LLM_MODEL_FILE}
# Forbid huggingface_hub from ever making a network call: it must use the
# models cached in the layers above, or fail loudly instead of downloading.
ENV HF_HUB_OFFLINE=1
VOLUME ["/data"]

EXPOSE 8000

# --limit-max-requests / body size is not capped by uvicorn itself; uploads
# are streamed to disk chunk by chunk (see app/storage.py), so there is no
# built-in file size limit here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
