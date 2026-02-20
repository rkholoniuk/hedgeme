# HedgeMe CI Runner Image
# Pre-built with TA-Lib, TensorFlow, and all Python dependencies
#
# Build locally (for testing):
#   docker build -t hedgeme-runner -f .github/docker/Dockerfile .
#
# The image is auto-built and pushed to ghcr.io when:
#   - requirements.txt changes
#   - This Dockerfile changes
#   - Manually triggered via Actions -> Build CI Runner Image

FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/rkholoniuk/hedgeme
LABEL org.opencontainers.image.description="HedgeMe ML Trading Bot CI Runner"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Verify installations
RUN python -c "import talib; print(f'TA-Lib version: {talib.__version__}')" \
    && python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" \
    && python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python", "--version"]
