FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY app.py .
COPY src/ ./src/   
COPY saved_model/ ./saved_model/
COPY templates/ ./templates/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]