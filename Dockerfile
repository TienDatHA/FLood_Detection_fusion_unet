# Multi-stage Dockerfile for Flood Detection System
# Base: TensorFlow GPU with GDAL/PROJ support

# ===============================================================================
# Stage 1: Base environment with system dependencies
# ===============================================================================
FROM tensorflow/tensorflow:2.15.0-gpu as base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    pkg-config \
    wget \
    curl \
    git \
    # GDAL and geospatial libraries
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Other dependencies
    libhdf5-dev \
    libnetcdf-dev \
    libopencv-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# ===============================================================================
# Stage 2: Python environment setup
# ===============================================================================
FROM base as python-env

# Upgrade pip and install wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install GDAL Python bindings (version must match system GDAL)
RUN pip install --no-cache-dir GDAL==$(gdal-config --version)

# Install core scientific packages
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.11.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.5 \
    seaborn==0.13.0 \
    plotly==5.17.0

# Install geospatial packages
RUN pip install --no-cache-dir \
    rasterio==1.3.9 \
    fiona==1.9.5 \
    shapely==2.0.2 \
    pyproj==3.6.1 \
    rtree==1.1.0

# Install deep learning and computer vision
RUN pip install --no-cache-dir \
    tensorflow-addons==0.23.0 \
    segmentation-models==1.0.1 \
    opencv-python==4.8.1.78 \
    Pillow==10.1.0 \
    albumentations==1.3.1 \
    imgaug==0.4.0

# Install additional ML tools
RUN pip install --no-cache-dir \
    tensorboard==2.15.1 \
    tqdm==4.66.1 \
    jupyter==1.0.0 \
    ipykernel==6.26.0 \
    notebook==7.0.6

# Install development tools
RUN pip install --no-cache-dir \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    click==8.1.7

# ===============================================================================
# Stage 3: Application setup
# ===============================================================================
FROM python-env as app

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data \
             /app/outputs \
             /app/models \
             /app/logs \
             /app/training_logs \
             /app/evaluation_logs \
             /app/inference_results \
             /app/visual_outputs \
             /app/dem_visualizations \
             /app/final_flood_results \
             /app/eval_results && \
    chown -R appuser:appuser /app

# Set environment variables for the application
ENV PYTHONPATH=/app
ENV DATA_ROOT=/app/data
ENV PROJECT_ROOT=/app

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Switch to non-root user
USER appuser

# Set default command
CMD ["python", "-c", "print('🚀 Flood Detection Docker Environment Ready!')"]

# ===============================================================================
# Development Stage (optional)
# ===============================================================================
FROM app as dev

USER root

# Install additional development tools
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    flake8==6.1.0 \
    black==23.11.0 \
    isort==5.12.0

# Install Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab==4.0.9 \
    jupyterlab-git==0.50.0

USER appuser

# ===============================================================================
# Production Stage (optimized)
# ===============================================================================
FROM app as prod

# Remove development dependencies and clean up
USER root
RUN pip uninstall -y jupyter notebook ipykernel && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER appuser

# Default command for production
CMD ["python", "inference_only.py"]