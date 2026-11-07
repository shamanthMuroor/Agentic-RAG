import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import app_graph, current_turn
from src.ingest import ensure_index
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=80)

class ChatResponse(BaseModel):
    answer: str
    thread_id: str

@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_index()
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

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or uuid.uuid4().hex
    try:
        result = app_graph.invoke(
            {"messages": [HumanMessage(content=request.question)], "tool_calls_count": 0},
            config={"configurable": {"thread_id": thread_id}},
        )
        turn = current_turn(result["messages"])
        answer = next((m.content for m in reversed(turn) if m.type == "ai"), None)
        if not answer:
            raise RuntimeError("The assistant returned no answer")
        return ChatResponse(answer=answer, thread_id=thread_id)
    except Exception as exc:
        logger.exception("Assistant request failed")
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable") from exc
