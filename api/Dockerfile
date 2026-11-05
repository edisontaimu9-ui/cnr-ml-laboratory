FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
# XGBoost needs the OpenMP runtime, which the slim image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY entrypoint.sh ./entrypoint.sh
RUN useradd -m appuser && mkdir -p /srv/data && chown -R appuser /srv && chmod +x /srv/entrypoint.sh
# Starts as root only to fix mounted-disk ownership; entrypoint.sh then drops to appuser.
EXPOSE 8000
# Render injects $PORT; default 8000 for local Docker
CMD ["/srv/entrypoint.sh"]
