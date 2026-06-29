from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# import lifespan และ router จาก dos_rag_08
from dos_rag_08 import lifespan as dos_lifespan, router as dos_router

# import app จาก api (ใช้ routes ตรงๆ)
import api as api_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    # รวม lifespan ของ dos_rag_08 (โหลด embedding model, reranker, DB pool, ollama)
    async with dos_lifespan(app):
        yield


app = FastAPI(
    title="DOS RAG Unified API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- routes จาก api.py (Gemini RAG) ---
app.add_api_route("/", api_module.root, methods=["GET"])
app.add_api_route("/ask", api_module.ask_endpoint, methods=["POST"])

# --- routes จาก dos_rag_08.py (Ollama RAG + OpenAI-compatible) ---
app.include_router(dos_router, prefix="/dos")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, workers=1)
