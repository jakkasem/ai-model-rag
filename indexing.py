import fitz  # PyMuPDF
import hashlib
import logging
import psycopg2
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("indexing")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_URL = os.getenv("DB_URL", "postgresql://myuser:mypassword@localhost:5432/mydb")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

client = ollama.Client(host=OLLAMA_HOST)


def compute_file_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


TABLE_NAME = "documents"


def is_already_indexed(cur, filename: str, file_hash: str) -> bool:
    cur.execute(
        "SELECT file_hash FROM public.pdf_index_log WHERE filename = %s AND table_name = %s",
        (filename, TABLE_NAME)
    )
    row = cur.fetchone()
    return row is not None and row[0] == file_hash


def update_index_log(cur, filename: str, file_hash: str) -> None:
    cur.execute(
        """
        INSERT INTO public.pdf_index_log (filename, table_name, file_hash)
        VALUES (%s, %s, %s)
        ON CONFLICT (filename, table_name) DO UPDATE
            SET file_hash = EXCLUDED.file_hash,
                indexed_at = now()
        """,
        (filename, TABLE_NAME, file_hash)
    )


def index_pdf(pdf_path: str):
    filename = os.path.basename(pdf_path)
    logger.info("Starting indexing: %s", filename)

    file_hash = compute_file_hash(pdf_path)
    logger.info("File hash: %s", file_hash)

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    try:
        if is_already_indexed(cur, filename, file_hash):
            logger.info("SKIP — %s is unchanged (hash match), no re-indexing needed.", filename)
            return

        logger.info("Hash changed or new file — starting full re-index for: %s", filename)

        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)
        logger.info("Extracted %d characters from PDF", len(full_text))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        logger.info("Total chunks: %d", len(chunks))

        cur.execute("DELETE FROM documents WHERE filename = %s", (filename,))
        logger.info("Deleted old data for: %s", filename)

        for i, chunk in enumerate(chunks):
            resp = client.embeddings(
                model=EMBED_MODEL,
                prompt=chunk,
                options={"keep_alive": "60m"}
            )
            embedding = resp["embedding"]
            cur.execute(
                "INSERT INTO documents (filename, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                (filename, i, chunk, embedding)
            )
            logger.info("Indexed chunk %d/%d", i + 1, len(chunks))

        update_index_log(cur, filename, file_hash)
        conn.commit()
        logger.info("Done — inserted %d chunks for: %s", len(chunks), filename)

    except Exception as e:
        conn.rollback()
        logger.exception("Error indexing %s: %s", filename, e)
        raise

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    pdf_file = os.getenv("PDF_FILE", "AnnualReport_2568.pdf")
    index_pdf(pdf_file)