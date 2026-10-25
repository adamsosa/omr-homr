FROM python:3.12-slim

LABEL org.opencontainers.image.title="omr-homr" \
      org.opencontainers.image.description="Isolated AGPL-3.0 OMR microservice wrapping homr CLI" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    patch \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install CPU-only PyTorch first to avoid 3 GB CUDA wheels, then pinned homr and FastAPI
ARG HOMR_VERSION=0.4.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision && \
    pip install --no-cache-dir \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.30.6" \
    "python-multipart==0.0.12" \
    "pillow==10.4.0" \
    "numpy==1.26.4" \
    "opencv-python-headless==4.10.0.84" \
    "homr==${HOMR_VERSION}" && \
    python -c "import homr"

COPY LICENSE README.md /app/
COPY omr_homr_service/ /app/omr_homr_service/

RUN chown -R appuser:appuser /app
USER appuser

# Pre-download homr checkpoints at build time so cold starts do not fetch from GitHub
RUN homr --init || homr --help >/dev/null

ARG GIT_COMMIT_SHA=unknown
ENV OMR_HOMR_COMMIT_SHA=${GIT_COMMIT_SHA}
ENV OMR_HOMR_REPO_URL="https://github.com/copianist/omr-homr"
ENV PORT=8081

EXPOSE 8081

CMD ["sh", "-c", "uvicorn omr_homr_service.app:app --host 0.0.0.0 --port ${PORT:-8081}"]
