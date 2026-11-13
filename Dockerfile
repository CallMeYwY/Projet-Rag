FROM python:3.12-slim AS base
WORKDIR /srv/rag
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Poppler rend les pages scannées en images. Tesseract lit ces images en français et en anglais.
RUN apt-get update && apt-get install -y --no-install-recommends \
      poppler-utils tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# La cible « tests » emporte les mêmes outils OCR que le service.
FROM base AS tests
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY app/ app/
COPY tests/ tests/
COPY demo/ demo/
CMD ["python", "-m", "pytest", "-q"]

FROM base AS api
COPY app/ app/
USER 65534:65534
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
