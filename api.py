import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)
PROFILE_PATH = Path(__file__).parent / "data" / "docs" / "profile.txt"
PROFILE_CHUNK_SIZE = 900

def _profile_chunks() -> list[str]:
    text = PROFILE_PATH.read_text(encoding="utf-8")
    return [text[i:i + PROFILE_CHUNK_SIZE] for i in range(0, len(text), PROFILE_CHUNK_SIZE)]

def _retrieve_profile(question: str, chunks: list[str]) -> str:
    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    ranked = sorted(
        chunks,
        key=lambda chunk: len(terms & set(re.findall(r"[a-z0-9]+", chunk.lower()))),
        reverse=True,
    )
    return "\n\n---\n\n".join(ranked[:3])

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=80)

class ChatResponse(BaseModel):
    answer: str
    thread_id: str

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not PROFILE_PATH.exists():
        raise RuntimeError(f"Missing profile document: {PROFILE_PATH}")
    app.state.profile_chunks = _profile_chunks()
    yield

app = FastAPI(title="Shamanth Portfolio Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Shamanth Portfolio Assistant", "status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or uuid.uuid4().hex
    try:
        context = _retrieve_profile(request.question, app.state.profile_chunks)
        answer = ChatOpenAI(
            base_url=settings.cloud_base_url,
            api_key=settings.groq_api_key,
            model=settings.cloud_model,
            temperature=0,
            max_tokens=settings.cloud_max_tokens,
            timeout=60,
            max_retries=2,
        ).invoke([
            SystemMessage(content=(
                "You are Shamanth's friendly portfolio representative. Answer only "
                "from the profile context. If the context does not answer the question, "
                "say that Shamanth has not provided the information. Do not guess or "
                "infer private details. Keep the answer under 120 words.\n\n"
                f"PROFILE CONTEXT:\n{context}"
            )),
            HumanMessage(content=request.question),
        ]).content
        return ChatResponse(answer=answer, thread_id=thread_id)
    except Exception as exc:
        logger.exception("Assistant request failed")
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable") from exc
