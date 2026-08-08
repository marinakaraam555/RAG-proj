from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Override number of retrieved chunks")


class RetrievedSource(BaseModel):
    source: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved: list[RetrievedSource] = []


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    num_chunks: int
