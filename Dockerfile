FROM python:3.11-slim

LABEL org.opencontainers.image.title="omr-homr" \
      org.opencontainers.image.description="Isolated AGPL-3.0 OMR microservice wrapping homr CLI" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    patch \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned homr and FastAPI HTTP wrapper dependencies
ARG HOMR_VERSION=0.4.0
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "numpy>=1.26.0" \
    "opencv-python-headless>=4.9.0" \
    "homr==${HOMR_VERSION}"

COPY LICENSE README.md /app/
COPY patches/ /app/patches/
COPY omr_homr_service/ /app/omr_homr_service/

# Apply minimal measure & staff bounding-box export patch to installed homr package
RUN SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])") && \
    if [ -d "${SITE_PKGS}/homr" ]; then \
      patch -p1 -d "${SITE_PKGS}" < /app/patches/export_measure_boxes.patch || true; \
    fi

ARG GIT_COMMIT_SHA=unknown
ENV OMR_HOMR_COMMIT_SHA=${GIT_COMMIT_SHA}
ENV OMR_HOMR_REPO_URL="https://github.com/copianist/omr-homr"
ENV PORT=8081

EXPOSE 8081

CMD ["uvicorn", "omr_homr_service.app:app", "--host", "0.0.0.0", "--port", "8081"]
