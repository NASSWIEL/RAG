"""HTTP API exposing the RAG engine for server deployment."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import RAGEngine

PDF_URL = os.environ.get("PDF_URL", "https://arxiv.org/pdf/2005.11401.pdf")

_state: dict = {"rag": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build or load the index once at startup, so requests only pay for retrieval."""
    _state["rag"] = RAGEngine(PDF_URL)
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


class Question(BaseModel):
    """Body of a query request."""

    question: str


@app.get("/health")
def health():
    """Report whether the index is loaded and the API can answer."""
    # RENDER_GIT_COMMIT lets the CD smoke test tell the new deploy from the old one.
    return {
        "status": "ok",
        "ready": _state["rag"] is not None,
        "commit": os.environ.get("RENDER_GIT_COMMIT", "unknown"),
    }


@app.post("/query")
def query(body: Question):
    """Answer a question about the indexed PDF."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return {"answer": _state["rag"].query(body.question)}
