# API-only mode: ComfyUI and ai-toolkit are host-native and out of scope for this container.
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend/dist frontend/dist
COPY config.example.json .
ENV LDS_DATA_DIR=/data \
    LDS_CONFIG=/data/config.json \
    LDS_HOST=0.0.0.0 \
    LDS_PORT=5050
EXPOSE 5050
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5050/api/health', timeout=3).read()"]
CMD ["python", "backend/run.py"]
