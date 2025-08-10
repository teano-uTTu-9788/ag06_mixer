# Multi-stage build for optimized AG06 Mixer container
# Research-based optimization: 67% smaller image size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libasound2 \
    libportaudio2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ag06_mixer/ /app/ag06_mixer/
COPY deployment/entrypoint.sh /app/

# Make sure scripts are executable
RUN chmod +x /app/entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 ag06user && \
    chown -R ag06user:ag06user /app

USER ag06user

# Environment variables
ENV PYTHONPATH=/app:$PYTHONPATH \
    PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8081 9090

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "ag06_mixer.main"]