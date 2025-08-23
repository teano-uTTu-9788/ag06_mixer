# AG06 Mixer - Production Docker Image
# MANU-Compliant Container Configuration

FROM python:3.11-slim

# Metadata
LABEL maintainer="AG06 Development Team"
LABEL version="2.0.0"
LABEL description="AG06 Mixer - Professional Audio Mixing Application"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 ag06user && \
    chown -R ag06user:ag06user /app

USER ag06user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "from ag06_manu_workflow import AG06WorkflowFactory; \
         import asyncio; \
         mgr = AG06WorkflowFactory.create_deployment_manager(); \
         health = asyncio.run(mgr.get_health_status()); \
         exit(0 if health.healthy else 1)"

# Expose ports
EXPOSE 8000 8080 9090

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV AG06_ENV=production
ENV AG06_LOG_LEVEL=INFO

# Run the application
CMD ["python3", "main.py"]