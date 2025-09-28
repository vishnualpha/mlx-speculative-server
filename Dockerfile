# MLX Speculative Server Dockerfile
# This Dockerfile is designed for Apple Silicon Macs with MLX support

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MLX_METAL_DEBUG=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY mlx_speculative/ ./mlx_speculative/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY run_tests.py pytest.ini ./
COPY setup.py pyproject.toml README.md LICENSE ./

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlx
RUN chown -R mlx:mlx /app
USER mlx

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "mlx_speculative.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
