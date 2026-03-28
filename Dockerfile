FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf

WORKDIR /app

# Install lightweight CPU-only PyTorch first to avoid huge GPU wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py ./main.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
