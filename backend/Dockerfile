FROM python:3.11-slim

# Keep output unbuffered and avoid writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# curl is useful for container debugging and (optionally) ECS container health checks.
# ca-certificates is required for outbound TLS and for downloading the RDS CA bundle.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Repo layout in this image:
#   /app/            (repo root)
#   /app/backend/    (FastAPI app, uvicorn entrypoint)
WORKDIR /app

# Install backend dependencies first for better layer caching.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy repo source. Large corpora/PDFs are excluded via .dockerignore.
COPY . /app

# Download the AWS RDS global CA bundle (used by sslmode=verify-full).
# Your DATABASE_URL should reference this path as sslrootcert=/etc/ssl/certs/rds-global-bundle.pem
ADD https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem /etc/ssl/certs/rds-global-bundle.pem

WORKDIR /app/backend

EXPOSE 8000

# "Start the API the same way as locally" (docs/RESEARCH_CONSOLE.md):
#   uvicorn app.main:app ...
# We allow PORT to be overridden by the platform, defaulting to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
