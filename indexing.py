import fitz  # PyMuPDF
import psycopg2
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_URL = os.getenv("DB_URL", "postgresql://myuser:mypassword@localhost:5432/mydb")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

client = ollama.Client(host=OLLAMA_HOST)

def index_pdf(pdf_path: str):
    # 1. อ่าน PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)

    # 2. Chunk — เพิ่มขนาดให้ครอบคลุมข้อมูลมากขึ้น
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(full_text)
    print(f"total chunks: {len(chunks)}")

    # 3. Clear ข้อมูลเก่าของไฟล์นี้ก่อน
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE filename = %s", (pdf_path,))
    print(f"cleared old data for {pdf_path}")

    # 4. Embed + Insert
    for i, chunk in enumerate(chunks):
        resp = client.embeddings(
            model=EMBED_MODEL,
            prompt=chunk,
            options={"keep_alive": "60m"}
        )
        embedding = resp["embedding"]
        cur.execute(
            "INSERT INTO documents (filename, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
            (pdf_path, i, chunk, embedding)
        )
        print(f"indexed chunk {i+1}/{len(chunks)}")

    conn.commit()
    cur.close()
    conn.close()
    print("done!")

if __name__ == "__main__":
    pdf_file = os.getenv("PDF_FILE", "AnnualReport_2568.pdf")
    index_pdf(pdf_file)