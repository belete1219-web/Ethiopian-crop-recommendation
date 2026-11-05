# Container for the Crop Recommendation API (prototype).
#   docker build -t crop-api:1.0 .
#   docker run --rm -p 8000:8000 crop-api:1.0          -> http://localhost:8000/docs
# To keep the prediction log outside the container:
#   docker run --rm -p 8000:8000 -v "$(pwd)/logs:/app/logs" crop-api:1.0
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY api.py input_checks.py model_metadata.json crop_recommendation_model.joblib ./
COPY outputs/model_training_features.csv ./outputs/model_training_features.csv

# Run as an unprivileged user; only the log directory is writable.
RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
