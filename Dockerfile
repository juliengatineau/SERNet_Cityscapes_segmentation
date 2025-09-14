# syntax=docker/dockerfile:1

########## Stage 1: builder (Poetry 2.x -> export -> venv) ##########
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Poetry aligné avec ton local + plugin export
RUN pip install --no-cache-dir "poetry==2.1.4" "poetry-plugin-export==1.8.0"

# Manifests d'abord (cache)
COPY pyproject.toml poetry.lock* /app/

# Export deps vers requirements.txt (pas de hashes)
RUN poetry export -f requirements.txt --output /tmp/reqs.txt --without-hashes

# Venv isolé pour runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Installe deps + gunicorn dans le venv
RUN pip install --no-cache-dir -r /tmp/reqs.txt && \
    pip install --no-cache-dir gunicorn && \
    python -c "import requests" || pip install --no-cache-dir requests

# (Option) bake du modèle public
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/model && \
    curl -L https://huggingface.co/juliengatineau/SERNet_cityscapes_trained/resolve/main/sernet_model.pt \
      -o /opt/model/sernet_model.pt

# Sanity check early-fail
RUN python - <<'PY'
import torch, torchvision, requests, sys
print("Python:", sys.version)
print("torch:", torch.__version__, "torchvision:", torchvision.__version__, "requests:", requests.__version__)
PY

########## Stage 2: runtime ##########
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"

# Venv complet (libs + binaires dont gunicorn)
COPY --from=builder /opt/venv /opt/venv

# Code + répertoires
COPY . /app
RUN mkdir -p static/images/pred static/images/uploads model

# Modèle baked (si utilisé)
COPY --from=builder /opt/model/sernet_model.pt /app/model/sernet_model.pt

ENV MODEL_REPO_ID="juliengatineau/SERNet_cityscapes_trained" \
    MODEL_FILENAME="sernet_model.pt" \
    PORT=8000

EXPOSE 8000
CMD ["gunicorn","--preload","-w","1","-k","gthread","-b","0.0.0.0:8000","app:app"]
