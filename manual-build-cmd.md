cd ai-model-rag

docker build -t ai-model-rag:latest .


docker run -d --name rag-api --env-file .env -p 8001:8001 ai-model-rag:latest