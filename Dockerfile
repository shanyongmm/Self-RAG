FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY rag ./rag
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY data ./data
COPY main.py pyproject.toml ./

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3)"

CMD ["sh", "-c", "python -m scripts.bootstrap && exec uvicorn main:app --host 0.0.0.0 --port 8001"]
