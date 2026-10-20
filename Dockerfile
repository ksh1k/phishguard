# Container for Render, Hugging Face Spaces or any host. Listens on $PORT (set by Render), default 7860.
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py app.py ./
COPY static ./static
COPY model ./model
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
