FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . /app/

# Create non-root user for security
RUN useradd -m -u 1000 agmixer && \
    chown -R agmixer:agmixer /app

USER agmixer

# Health check using our health_check.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python health_check.py || exit 1

# Default command runs the web app
CMD ["python", "web_app.py"]