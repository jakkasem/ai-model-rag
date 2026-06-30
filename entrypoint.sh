#!/bin/sh
set -e

echo "[entrypoint] Running indexing.py..."
python indexing.py

echo "[entrypoint] Running readPDFInsertData_04.py..."
python readPDFInsertData_04.py

echo "[entrypoint] Starting Unified API..."
exec uvicorn main:app --host 0.0.0.0 --port 8001
