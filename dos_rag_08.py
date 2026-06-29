# dos_rag_v8.py – Speed-optimized RAG API (Round 3)
# Speed fixes vs v7:
# - ใช้ re.findall() แทน list comprehension ในการนับตัวอักษรจีน (ลด CPU bound / เร็วขึ้นหลายเท่า)
# - เปลี่ยนการเช็กคิวรีซ้ำในดึงข้อมูล Multi-query จากข้อความ (String slice) มาใช้ Unique Identifier/Content hashing เพื่อลดการวนลูปเช็กสตริง
# - ปรับปรุงโครงสร้าง Context Manager (with pool.getconn()) เพื่อป้องกัน Connection Leak และกำจัดสภาวะเสี่ยง AttributeError ในบล็อก finally

import asyncio
import hashlib
import logging
import time
import re
from contextlib import asynccontextmanager
from collections import defaultdict, Counter
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import Client
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("dos_rag")

# ---------------------------------------------------------------------------
# Config (ดึงจาก env vars ทั้งหมด — ไม่ hardcode)
# ---------------------------------------------------------------------------
OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

DB_HOST            = os.getenv("DB_HOST", "localhost")
DB_NAME            = os.getenv("DB_NAME", "mydb")
DB_USER            = os.getenv("DB_USER", "postgres")
DB_PASSWORD        = os.getenv("DB_PASSWORD", "postgres")
DB_PORT            = int(os.getenv("DB_PORT", "5432"))

# --- Threshold configs ---
VECTOR_DISTANCE_THRESHOLD   = float(os.getenv("VECTOR_DISTANCE_THRESHOLD", "0.45"))
RERANKER_SCORE_THRESHOLD    = float(os.getenv("RERANKER_SCORE_THRESHOLD", "0.3"))
VECTOR_TOP_K                = int(os.getenv("VECTOR_TOP_K", "3"))
RERANK_TOP_N                = int(os.getenv("RERANK_TOP_N", "5"))
QUERIES_PER_REQUEST         = int(os.getenv("QUERIES_PER_REQUEST", "2"))

# ---------------------------------------------------------------------------
# Singletons (โหลดครั้งเดียวตอน startup)
# ---------------------------------------------------------------------------
_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_db_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_ollama_client: Optional[Client] = None

# Response cache — คำถามเดิมที่ถามซ้ำใน TTL จะตอบทันทีไม่ต้องรันใหม่
_RESPONSE_CACHE: dict[str, tuple[str, float]] = {}
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

def _cache_get(key: str) -> Optional[str]:
    if key in _RESPONSE_CACHE:
        answer, ts = _RESPONSE_CACHE[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            logger.info("Cache HIT")
            return answer
        del _RESPONSE_CACHE[key]
    return None

def _cache_set(key: str, answer: str) -> None:
    _RESPONSE_CACHE[key] = (answer, time.time())
    if len(_RESPONSE_CACHE) > 500:
        cutoff = time.time() - CACHE_TTL_SECONDS
        for k in [k for k, (_, ts) in _RESPONSE_CACHE.items() if ts < cutoff]:
            del _RESPONSE_CACHE[k]

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model …")
        _embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
    return _embed_model

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model …")
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker

def get_db_pool() -> pg_pool.ThreadedConnectionPool:
    global _db_pool
    if _db_pool is None:
        logger.info("Creating DB connection pool …")
        _db_pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
        )
    return _db_pool

def get_ollama() -> Client:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = Client(host=OLLAMA_HOST)
    return _ollama_client

# ---------------------------------------------------------------------------
# Lifespan (FastAPI >= 0.93)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(get_embed_model)
    await asyncio.to_thread(get_reranker)
    await asyncio.to_thread(get_db_pool)
    get_ollama()
    logger.info("✅ All models and DB pool ready.")
    yield
    if _db_pool:
        _db_pool.closeall()
        logger.info("DB pool closed.")

# ---------------------------------------------------------------------------
# Router (เปลี่ยนจาก standalone app เป็น router เพื่อให้ main.py mount ได้)
# ---------------------------------------------------------------------------
router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    messages: list[Message]

class ChatRequest(BaseModel):
    message: str
    history: list[list[str]] = []

class ChatResponse(BaseModel):
    answer: str
    history: list[list[str]]

# ---------------------------------------------------------------------------
# Helper: Translate & Expand
# ---------------------------------------------------------------------------
def _translate_to_english(text: str, cache: dict) -> str:
    if text in cache:
        return cache[text]
    cache[text] = text
    return text

_HR_SYNONYMS: dict[str, list[str]] = {
    "ทดลองงาน":    ["probationary period", "probation evaluation days"],
    "ลา":          ["leave policy", "employee leave entitlement"],
    "เงินเดือน":   ["salary compensation", "basic pay rate"],
    "ค่าแท็กซี่":  ["taxi fare reimbursement", "transportation claim"],
    "ประกัน":      ["insurance policy", "accident coverage"],
    "ล่วงเวลา":    ["overtime pay", "extra hours work"],
    "ลาออก":       ["resignation", "termination policy"],
    "probation":   ["probationary period", "trial period employee"],
    "leave":       ["employee leave policy", "days off work"],
}

def _expand_queries(user_message: str, english_query: str) -> list[str]:
    combined = (user_message + " " + english_query).lower()
    extras: list[str] = []
    for keyword, synonyms in _HR_SYNONYMS.items():
        if keyword.lower() in combined:
            extras.extend(synonyms)
            break

    all_queries = [english_query] + [q for q in extras if q != english_query]
    seen: set[str] = set()
    result: list[str] = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            result.append(q)
    return result[:QUERIES_PER_REQUEST]

# ---------------------------------------------------------------------------
# Core RAG logic (sync)
# ---------------------------------------------------------------------------
def _run_rag_sync(user_message: str, history: list[Message]) -> str:
    translate_cache: dict[str, str] = {}
    pool = get_db_pool()
    conn = None

    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            english_query = _translate_to_english(user_message, translate_cache)
            expanded_queries = _expand_queries(user_message, english_query)
            logger.info("Using %d queries for vector search", len(expanded_queries))

            embed_model = get_embed_model()
            query_texts = [f"query: {q}" for q in expanded_queries]
            batch_embeddings = embed_model.encode(query_texts, batch_size=8).tolist()

            seen_content_hashes: set[str] = set()
            all_rows: list[tuple] = []
            
            for q_embedding in batch_embeddings:
                cur.execute(
                    """
                    SELECT content, title, (embedding <=> %s::public.vector) AS distance
                    FROM public.fujitsu_hr_policy
                    ORDER BY distance ASC
                    LIMIT %s;
                    """,
                    (q_embedding, VECTOR_TOP_K),
                )
                for row in cur.fetchall():
                    content = row[0]
                    logger.info("Vector content found %s", row[0])
                    # ใช้ MD5 Hash แทนการสไลซ์สตริงยาวๆ ป้องกันปัญหา CPU-bound overhead และแม่นยำกว่า
                    c_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    if c_hash not in seen_content_hashes:
                        seen_content_hashes.add(c_hash)
                        all_rows.append(row)

            if not all_rows:
                logger.warning("No rows returned from vector search.")
                return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในระบบฐานข้อมูลนโยบายครับ"

            filtered_rows = [r for r in all_rows if r[2] <= VECTOR_DISTANCE_THRESHOLD]
            logger.info(
                "Multi-query vector search: %d total unique rows, %d after distance threshold (<=%.2f)",
                len(all_rows), len(filtered_rows), VECTOR_DISTANCE_THRESHOLD,
            )
            if not filtered_rows:
                logger.warning("All rows exceed distance threshold — relaxing to top-10.")
                filtered_rows = sorted(all_rows, key=lambda r: r[2])[:10]

            # --- Rerank ---
            reranker = get_reranker()
            all_candidate_chunks = [r[0] for r in filtered_rows]
            all_candidate_titles = [r[1] for r in filtered_rows]

            pairs = [[user_message, chunk] for chunk in all_candidate_chunks]
            scores = reranker.predict(pairs)

            scored_with_title = sorted(
                zip(scores, all_candidate_chunks, all_candidate_titles),
                key=lambda x: x[0],
                reverse=True,
            )

            title_votes = Counter(t for _, _, t in scored_with_title[:3] if t)
            final_title = title_votes.most_common(1)[0][0] if title_votes else all_candidate_titles[0]
            logger.info("Title selected by reranker top-3 vote: %r", final_title)

            top_score = scored_with_title[0][0] if scored_with_title else 0
            existing_title_chunks = {c for _, c, t in scored_with_title if t == final_title}
            HIGH_CONFIDENCE_SCORE = 0.7
            need_more = (
                len(existing_title_chunks) < RERANK_TOP_N
                and top_score < HIGH_CONFIDENCE_SCORE
            )
            if need_more:
                cur.execute(
                    "SELECT content FROM public.fujitsu_hr_policy WHERE title = %s LIMIT 30;",
                    (final_title,),
                )
                db_title_chunks = [r[0] for r in cur.fetchall()]
                new_chunks = [c for c in db_title_chunks if c not in existing_title_chunks]
                if new_chunks:
                    new_scores = reranker.predict([[user_message, c] for c in new_chunks])
                    for s, c in zip(new_scores, new_chunks):
                        scored_with_title.append((s, c, final_title))
                    scored_with_title.sort(key=lambda x: x[0], reverse=True)
                    logger.info("Reranked %d additional chunks from title", len(new_chunks))

            title_scored = [(s, c) for s, c, t in scored_with_title if t == final_title]
            above_threshold = [(s, c) for s, c in title_scored if s >= RERANKER_SCORE_THRESHOLD]
            logger.info(
                "Final: %d title-chunks, %d above threshold (>=%.2f), top score: %.3f",
                len(title_scored), len(above_threshold), RERANKER_SCORE_THRESHOLD,
                title_scored[0][0] if title_scored else 0,
            )
            if not above_threshold:
                logger.warning("All chunks below threshold — using top-1 fallback.")
                above_threshold = title_scored[:1]

            top_chunks = [c for _, c in above_threshold[:RERANK_TOP_N]]
            context_str = "\n\n".join(top_chunks)

    except Exception as exc:
        logger.exception("DB/model error in _run_rag_sync")
        return f"เกิดข้อผิดพลาดในระบบ: {exc}"
    finally:
        if conn:
            pool.putconn(conn)

    # --- ส่งให้ LLM ตอบ ---
    system_prompt = (
        "คุณคือผู้ช่วย HR สำหรับบริษัท Fujitsu/AEONTS "
        "ตอบคำถามโดยอ้างอิงจาก Context ที่ให้มาเท่านั้น อย่าแต่งเติมข้อมูล "
        "ถ้า Context ไม่มีคำตอบให้ตอบว่า: "
        "'ขออภัย ข้อมูลที่ฉันมีไม่ครอบคลุมคำถามนี้ กรุณาติดต่อฝ่าย HR โดยตรงครับ' "
        "CRITICAL INSTRUCTION: You MUST respond in Thai language ONLY. "
        "DO NOT use Chinese, English, or any other language. "
        "Even if the context documents are in Chinese or English, "
        "your answer MUST be written entirely in Thai. "
        "กฎเหล็ก: ตอบเป็นภาษาไทยเท่านั้น ห้ามตอบภาษาอื่นเด็ดขาด "
        "ตอบให้ตรงประเด็น ชัดเจน และเป็นมืออาชีพ"
    )

    messages_payload = [{"role": "system", "content": system_prompt}]
    recent_history = history[-4:] if len(history) > 4 else history
    for msg in recent_history:
        messages_payload.append({"role": msg.role, "content": msg.content})

    messages_payload.append({
        "role": "user",
        "content": (
            f"[เอกสารอ้างอิง: {final_title}]\n"
            f"--- เนื้อหาเอกสาร ---\n{context_str}\n"
            f"--- สิ้นสุดเนื้อหา ---\n\n"
            f"คำถาม: {user_message}\n\n"
            f"คำแนะนำ: อ่านเนื้อหาเอกสารด้านบนอย่างละเอียด แล้วตอบคำถามเป็นภาษาไทยเท่านั้น "
            f"ห้ามใช้ภาษาจีนหรือภาษาอังกฤษในคำตอบ "
            f"โดยระบุตัวเลข/ข้อมูลที่ชัดเจนจากเอกสาร\nคำตอบ (ภาษาไทยเท่านั้น):"
        ),
    })

    try:
        resp = get_ollama().chat(
            model=OLLAMA_MODEL,
            messages=messages_payload,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 400,
                "repeat_penalty": 1.15,
            },
        )
        answer = resp["message"]["content"]

        # ปรับปรุงความเร็ว: ใช้ Regex แทน list comprehension วนลูปเช็กอักษรจีนทีละตัว
        chinese_chars = len(re.findall(r"[一-鿿]", answer))
        if chinese_chars > len(answer) * 0.3:
            logger.warning("Response detected as Chinese, retrying with stronger prompt...")
            messages_payload[-1]["content"] += (
                "\n\n[IMPORTANT: Your previous response was in Chinese. "
                "You MUST respond in THAI ONLY.]"
            )
            resp2 = get_ollama().chat(
                model=OLLAMA_MODEL,
                messages=messages_payload,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 300,
                    "repeat_penalty": 1.15,
                },
            )
            answer = resp2["message"]["content"]

        return answer
    
    except Exception as exc:
        logger.exception("Ollama chat error")
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อ LLM: {exc}"

# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------
def _normalize_for_cache(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    return normalized.rstrip("?!.ๆฯ ")

async def run_rag(user_message: str, history: list[Message]) -> str:
    cache_key = hashlib.md5(_normalize_for_cache(user_message).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        return cached
    answer = await asyncio.to_thread(_run_rag_sync, user_message, history)
    _cache_set(cache_key, answer)
    return answer

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@router.get("/v1/api/config")
async def get_config():
    return {
        "status": True,
        "name": "DOS RAG API",
        "version": "8.0.0",
        "default_models": "dos_rag",
        "default_prompt_suggestions": [],
    }

@router.get("/v1/models")
@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "dos_rag",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "system",
        }],
    }

@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def openai_chat(req: OpenAIChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
    user_message = req.messages[-1].content
    history = list(req.messages[:-1])
    ai_answer = await run_rag(user_message, history)
    ts = int(time.time())
    return {
        "id": f"chatcmpl-{ts}",
        "object": "chat.completion",
        "created": ts,
        "model": "dos_rag",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ai_answer},
            "finish_reason": "stop",
        }],
    }

@router.post("/chat", response_model=ChatResponse)
async def legacy_chat(request: ChatRequest):
    formatted_history: list[Message] = []
    for turn in request.history:
        if len(turn) == 2:
            formatted_history.append(Message(role="user", content=turn[0]))
            formatted_history.append(Message(role="assistant", content=turn[1]))
    ai_answer = await run_rag(request.message, formatted_history)
    new_history = request.history + [[request.message, ai_answer]]
    return ChatResponse(answer=ai_answer, history=new_history)

@router.post("/v1/cache/clear")
async def clear_cache():
    count = len(_RESPONSE_CACHE)
    _RESPONSE_CACHE.clear()
    return {"cleared": count}

@router.post("/v1/debug/search")
async def debug_search(req: BaseModel):
    # ยึดโครงสร้างเดิมไว้สำหรับตรวจสอบพฤติกรรม RAG
    pass

if __name__ == "__main__":
    import uvicorn
    # standalone mode — สร้าง app ชั่วคราวสำหรับ dev/test เท่านั้น
    _standalone = FastAPI(lifespan=lifespan)
    _standalone.include_router(router)
    uvicorn.run(_standalone, host="0.0.0.0", port=8000)