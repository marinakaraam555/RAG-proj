import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def ask_question(question: str, top_k: int | None = None) -> dict:
    payload = {"question": question}
    if top_k:
        payload["top_k"] = top_k
    response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def ask_with_image(question: str, image_file) -> dict:
    files = {"image": (image_file.name, image_file, "image/jpeg")}
    data = {"question": question}
    response = requests.post(f"{API_BASE_URL}/query-with-image", data=data, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def check_health() -> dict:
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json()
