# Base image with Python. slim keeps the image small.
FROM python:3.11-slim

# System libraries required by OpenCV, EasyOCR, and PaddleOCR.
# libgomp1 is needed by PaddlePaddle's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements_app.txt .
# Install the CPU-only build of torch to avoid pulling large CUDA wheels.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements_app.txt

# Copy the application code and the trained model.
COPY app/ ./app/
COPY best.pt .

# API port (FastAPI/uvicorn) and UI port (Streamlit).
EXPOSE 8000 8501

# Default command is overridden per-service in docker-compose.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
