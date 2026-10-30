# ── Stage 1: build dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS base

# System libs required by OpenCV and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: app ────────────────────────────────────────────────────────────
WORKDIR /app

# Install Python dependencies first (layer-cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artefacts
COPY app.py        .
COPY prediction.py .
COPY models/       models/
COPY templates/    templates/
COPY static/       static/

# Flask runtime settings
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120 app:app"]
