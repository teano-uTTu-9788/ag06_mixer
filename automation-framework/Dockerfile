# Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System deps for gevent and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev libssl-dev libportaudio2 curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY fixed_ai_mixer.py /app/fixed_ai_mixer.py

EXPOSE 8080
ENV OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
CMD ["gunicorn","-w","1","-k","gevent","-b","0.0.0.0:8080","fixed_ai_mixer:app"]