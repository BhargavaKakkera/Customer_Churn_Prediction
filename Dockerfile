FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by LightGBM/XGBoost C extensions (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .
COPY app.py .
COPY src/ ./src/   
COPY saved_model/ ./saved_model/
COPY templates/ ./templates/
COPY static/ ./static/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]