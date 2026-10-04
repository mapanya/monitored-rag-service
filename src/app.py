"""The web service. This is what you (or any program) talk to.

Start it with:

    uvicorn src.app:app --port 8000

Then open http://localhost:8000 in your browser. You land on a built-in
test page where you can try the /ask endpoint without writing any code:
click POST /ask, then "Try it out", type a question and click Execute.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.observability import setup_tracing
from src.rag_pipeline import RAGPipeline

# Created once when the service starts, reused for every request.
_pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs at startup: connect tracing first so even the first question
    # is recorded, then open the pipeline's connections.
    setup_tracing()
    _pipeline["rag"] = RAGPipeline()
    yield
    # Runs at shutdown: close the database connection politely.
    _pipeline["rag"].close()


app = FastAPI(
    title="Monitored RAG Service",
    description=(
        "Ask questions about the loaded documents. Every request is "
        "checked by a security guardrail and recorded in the Phoenix "
        "dashboard at http://localhost:6006."
    ),
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Your question")


class AskResponse(BaseModel):
    blocked: bool
    answer: str
    sources: list[str]
    passages_flagged: int


@app.get("/", include_in_schema=False)
def home():
    """Opening the service in a browser lands on the test page."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Quick check that the service is up."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answers a question using the loaded documents."""
    result = _pipeline["rag"].answer(request.question)
    return AskResponse(
        blocked=result["blocked"],
        answer=result["answer"],
        sources=result["sources"],
        passages_flagged=result["passages_flagged"],
    )
