CREATE EXTENSION IF NOT EXISTS vector;

drop TABLE documents;
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  filename TEXT,
  chunk_index INT,
  content TEXT,
  embedding vector(1024)
);
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);


drop TABLE fujitsu_hr_policy;
CREATE TABLE fujitsu_hr_policy (
  id INT(4) PRIMARY KEY,
  title Varchar(255),
  content TEXT,
  embedding vector(1024)
);