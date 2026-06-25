
## ================================================================================================================---
## Install cloudflared & start run
## ================================================================================================================---
cloudflared tunnel --url http://localhost:11434 --http-host-header="localhost:11434" --tunnel-timeout 300s






## ================================================================================================================---
## Install open-webui
## ================================================================================================================---

py -3.11 -m pip install open-webui

%USERPROFILE%\AppData\Local\Python\pythoncore-3.11-64\Scripts\open-webui.exe serve


%USERPROFILE%\AppData\Local\Python\pythoncore-3.11-64\Scripts\open-webui.exe serve --port 8090


goto http://localhost:8090/

user: admin
email: admin@admin.com
pass: P@ssw0rd




## ================================================================================================================---
## Python RAG Pipeline
## ติดตั้ง dependencies:
## ================================================================================================================---
$ pip install pymupdf psycopg2-binary ollama langchain-text-splitters


1. docker compose up -d        → postgres พร้อม
2. ollama pull qwen3-embedding:0.6b  → model พร้อม
3. python indexing.py          → อ่าน PDF → embed → เก็บใน pgvector
4. python query.py             → ถามตอบได้

## ================================================================================================================---
## Call via api with cmd
## ================================================================================================================---
curl http://localhost:11434/api/embed   -d '{"model": "qwen3-embedding:0.6b", "input": "ทดสอบภาษาไทย"}'

curl http://localhost:11434/api/embed -d "{\"model\": \"qwen3-embedding:0.6b\", \"input\": \"ทดสอบภาษาไทย\"}"


curl https://pavilion-choosing-tribute-bracket.trycloudflare.com/api/embed   -d '{"model": "qwen3:1.7b", "input": "Requirement prioritization มีอะไรบ้าง"}'

curl https://pavilion-choosing-tribute-bracket.trycloudflare.com/api/embed  -d "{\"model\": \"qwen3:1.7b\", \"input\": \"Requirement prioritization มีอะไรบ้าง\"}"

curl -X POST https://pavilion-choosing-tribute-bracket.trycloudflare.com/api/chat -d "{\"model\": \"qwen3:1.7b\", \"messages\": [{\"role\": \"user\", \"content\": \"Requirement prioritization มีอะไรบ้าง\"}]}"
curl -X POST https://pavilion-choosing-tribute-bracket.trycloudflare.com/api/chat -d "{\"model\": \"qwen3:1.7b\", \"messages\": [{\"role\": \"user\", \"content\": \"Requirement prioritization มีอะไรบ้าง\"}], \"stream\": false}"


curl -s -X POST https://pavilion-choosing-tribute-bracket.trycloudflare.com/api/chat -d "{\"model\": \"qwen3:1.7b\", \"messages\": [{\"role\": \"user\", \"content\": \"Requirement prioritization มีอะไรบ้าง\"}]}" | findstr /C:"\"content\":"





## ================================================================================================================---
## start fastapi with cmd
## ================================================================================================================---

api.py -- Create
# Install
pip install fastapi uvicorn

# Start Run API
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload


curl -X POST http://localhost:8000/ask -H "Content-Type: application/json"  -d "{\"question\": \"Requirement prioritization มีอะไรบ้าง\"}"

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json"  -d "{\"question\": \"ระบบ support active user at least เท่าไหร่\"}"





------- Gemini on cloudflared
pip install google-genai

pip install google-generativeai

แก้ query.py