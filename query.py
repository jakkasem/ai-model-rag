import time
import psycopg2
import ollama
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DB_URL = os.getenv("DB_URL", "postgresql://myuser:mypassword@localhost:5432/mydb")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ollama_client = ollama.Client(host=OLLAMA_HOST)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def search(question: str, top_k: int = 5) -> list[str]:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    t0 = time.time()
    resp = ollama_client.embeddings(
        model=EMBED_MODEL,
        prompt=question,
        options={"keep_alive": "60m"}
    )
    q_embedding = resp["embedding"]
    print(f"[timing] embedding:       {time.time()-t0:.2f}s")

    t1 = time.time()
    cur.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) AS score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (q_embedding, q_embedding, top_k))
    results = cur.fetchall()
    print(f"[timing] pgvector search: {time.time()-t1:.2f}s")

    cur.close()
    conn.close()
    return [row[0] for row in results]

def ask(question: str) -> str:
    contexts = search(question)
    context_text = "\n\n".join(contexts)

    prompt = f"""ตอบคำถามเป็นภาษาไทยเท่านั้น จาก context ที่ให้มาเท่านั้น ถ้าไม่มีข้อมูลให้บอกว่าไม่ทราบ

Context:
{context_text}

คำถาม: {question}
คำตอบ (ภาษาไทยเท่านั้น):"""

    t2 = time.time()
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    print(f"[timing] LLM generate:    {time.time()-t2:.2f}s")

    return resp.text

if __name__ == "__main__":
    while True:
        q = input("\nถามว่า: ")
        print(ask(q))