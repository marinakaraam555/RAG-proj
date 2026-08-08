from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.query import router as query_router
from app.services.retrieval import get_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load vector store + embedding model once at startup, not per-request
    get_retriever()
    yield


app = FastAPI(
    title="Traffic Law RAG Assistant API",
    description="RAG-powered Q&A over traffic-law and vehicle-regulation documents, "
                 "with an optional YOLO vision component for uploaded vehicle/plate images.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:8501", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/")
def root():
    return {"message": "Traffic Law RAG Assistant API — see /docs for Swagger UI"}
