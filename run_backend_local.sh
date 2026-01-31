#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
VENV_DIR="${BACKEND_DIR}/.venv"

if [[ ! -d "${BACKEND_DIR}" ]]; then
  echo "backend directory not found: ${BACKEND_DIR}"
  exit 1
fi

cd "${BACKEND_DIR}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install -r requirements.txt

if [[ -z "${OLLAMA_MODEL:-}" ]]; then
  export OLLAMA_MODEL="gemini-3-flash-preview:cloud"
fi

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama model: ${OLLAMA_MODEL}"
  echo "Tip: if the model isn't pulled yet, run: ollama pull ${OLLAMA_MODEL}"
else
  echo "Warning: 'ollama' not found. Install Ollama and run: ollama pull ${OLLAMA_MODEL}"
fi

echo "Local backend: http://127.0.0.1:8000/api/chat"
exec "${VENV_DIR}/bin/uvicorn" main:app --host 127.0.0.1 --port 8000 --reload
