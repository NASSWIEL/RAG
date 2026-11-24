FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch keeps the image small enough for a free-tier server (no CUDA libraries).
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY *.py ./

# Render injects PORT; 8000 is the local default.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
