#!/bin/bash
set -euo pipefail

# Usage:
#   SSH_TARGET=ubuntu@106.54.188.236 ./deploy_prod.sh
#
# One-time server prep is documented in `deploy/README.md`.

SSH_TARGET="${SSH_TARGET:-ubuntu@106.54.188.236}"
REMOTE_WEB_ROOT="${REMOTE_WEB_ROOT:-/var/www/huaodong.online}"
REMOTE_BACKEND_ROOT="${REMOTE_BACKEND_ROOT:-/opt/huaodong-chatbot}"
SERVICE_NAME="${SERVICE_NAME:-huaodong-chatbot}"

echo "Deploying static site -> ${SSH_TARGET}:${REMOTE_WEB_ROOT}"
ssh "${SSH_TARGET}" "mkdir -p '${REMOTE_WEB_ROOT}' '${REMOTE_BACKEND_ROOT}/backend'"

rsync -avz --delete \
  --exclude '.git' \
  --exclude '.github' \
  --exclude 'node_modules' \
  --exclude 'backend' \
  --exclude 'deploy' \
  --exclude '*.sh' \
  --exclude 'package.json' \
  --exclude 'package-lock.json' \
  ./ "${SSH_TARGET}:${REMOTE_WEB_ROOT}/"

echo "Deploying backend -> ${SSH_TARGET}:${REMOTE_BACKEND_ROOT}/backend"
rsync -avz --delete \
  --exclude 'backend/.venv' \
  backend/ "${SSH_TARGET}:${REMOTE_BACKEND_ROOT}/backend/"

echo "Installing backend deps (venv) on server..."
ssh "${SSH_TARGET}" "cd '${REMOTE_BACKEND_ROOT}/backend' && \
  python3 -m venv .venv && \
  ./.venv/bin/pip install -r requirements.txt"

echo "Restarting backend..."
ssh "${SSH_TARGET}" "set -e; \
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^${SERVICE_NAME}\\.service'; then \
    sudo systemctl restart '${SERVICE_NAME}'; \
  else \
    pkill -f 'uvicorn main:app' || true; \
    KB_LOCAL_ROOT='${REMOTE_WEB_ROOT}' \
    KB_INCLUDE_GITHUB='1' \
    OLLAMA_MODEL='gemini-3-flash-preview:cloud' \
    nohup '${REMOTE_BACKEND_ROOT}/backend/.venv/bin/python' -m uvicorn main:app --host 127.0.0.1 --port 8000 > '${REMOTE_BACKEND_ROOT}/backend/backend.log' 2>&1 & \
    echo 'Started backend via nohup (install systemd service for production).'; \
  fi"

echo "Done."
echo "Health check: https://huaodong.online/api/health"

