#!/bin/sh
# Mounted disks (Render) arrive root-owned. Fix ownership, then drop to the unprivileged user.
set -e
mkdir -p /srv/data
if [ "$(id -u)" = "0" ]; then
  chown -R appuser:appuser /srv/data || true
  exec setpriv --reuid=appuser --regid=appuser --init-groups \
    uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
