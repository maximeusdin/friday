FROM python:3.11-slim

# Keep Python output unbuffered and avoid .pyc writes
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build deps (psycopg2-binary doesn't need libpq dev headers)
RUN pip install --no-cache-dir --upgrade pip

# Copy repo
WORKDIR /app
COPY . /app

# Install backend deps
RUN pip install --no-cache-dir -r backend/requirements.txt

# Run the API from backend/ so `import app...` works
WORKDIR /app/backend

# Optional: set at build time with --build-arg GIT_SHA=...
ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA}

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

