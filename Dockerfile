FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    tesseract-ocr \
    tesseract-ocr-tha \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py indexing.py query.py query_local.py init.sql ./
COPY dos_rag_08.py readPDFInsertData_04.py main.py ./
COPY ai-booklet.pdf .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000 8001

ENTRYPOINT ["./entrypoint.sh"]
