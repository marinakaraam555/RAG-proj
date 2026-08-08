"""Builds the grounded prompt and calls the local Ollama LLM."""
import ollama

from app.core.config import settings


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    return f"""You are a traffic-law assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't have enough information.
Cite the source file name for every claim you make, in the form (Source: filename).

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, hits: list[dict], image_context: str | None = None) -> str:
    prompt = build_prompt(question, hits)
    if image_context:
        prompt += f"\n\n[Image context]: {image_context}"

    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
