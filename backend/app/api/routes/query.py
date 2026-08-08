import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Form

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, RetrievedSource
from app.services.retrieval import get_retriever
from app.services.generation import generate_answer

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        retriever = get_retriever()
        return HealthResponse(
            status="ok", vector_store_loaded=True, num_chunks=retriever.num_chunks
        )
    except Exception:
        return HealthResponse(status="degraded", vector_store_loaded=False, num_chunks=0)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    retriever = get_retriever()
    hits = retriever.retrieve(request.question, top_k=request.top_k)
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant context found")

    answer = generate_answer(request.question, hits)
    sources = sorted({h["source"] for h in hits})
    retrieved = [
        RetrievedSource(source=h["source"], snippet=h["text"][:200], score=round(h["score"], 3))
        for h in hits
    ]
    return QueryResponse(answer=answer, sources=sources, retrieved=retrieved)


@router.post("/query-with-image", response_model=QueryResponse)
async def query_with_image(question: str = Form(...), image: UploadFile = File(...)):
    """Extended Track endpoint: accepts a question + an image, fuses YOLO detection
    output into the RAG prompt context alongside the retrieved text chunks.
    """
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    from app.services.vision import detect_image_context

    with tempfile.NamedTemporaryFile(suffix=Path(image.filename).suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        image_context = detect_image_context(tmp_path)
    except Exception as e:
        image_context = f"(vision model unavailable: {e})"
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    retriever = get_retriever()
    hits = retriever.retrieve(question)
    answer = generate_answer(question, hits, image_context=image_context)
    sources = sorted({h["source"] for h in hits})
    retrieved = [
        RetrievedSource(source=h["source"], snippet=h["text"][:200], score=round(h["score"], 3))
        for h in hits
    ]
    return QueryResponse(answer=f"{answer}\n\n[Image analysis: {image_context}]", sources=sources, retrieved=retrieved)
