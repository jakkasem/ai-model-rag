CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS vector;


drop TABLE documents;

CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  filename TEXT,
  chunk_index INT,
  content TEXT,
  embedding vector(1024)  -- แก้จาก 1536 → 1024
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);