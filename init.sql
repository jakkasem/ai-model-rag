CREATE EXTENSION IF NOT EXISTS vector;



CREATE TABLE IF NOT EXISTS public.pdf_index_log (
    filename TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT now()
);
-- เพิ่ม column table_name
ALTER TABLE public.pdf_index_log
    ADD COLUMN IF NOT EXISTS table_name TEXT NOT NULL DEFAULT '';

-- ลบ primary key เดิม (filename อย่างเดียว) แล้วสร้างใหม่เป็น composite
ALTER TABLE public.pdf_index_log
    DROP CONSTRAINT IF EXISTS pdf_index_log_pkey;

ALTER TABLE public.pdf_index_log
    ADD PRIMARY KEY (filename, table_name);



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