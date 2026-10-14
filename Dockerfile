# Container for Hugging Face Spaces (Docker SDK) or any host. Serves the app on port 7860.
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py app.py ./
COPY static ./static
COPY model ./model
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
