# API-only mode: ComfyUI and ai-toolkit are host-native and out of scope for this container.
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend/dist frontend/dist
COPY config.example.json .
ENV LDS_DATA_DIR=/data
EXPOSE 5000
CMD ["python", "backend/run.py"]
