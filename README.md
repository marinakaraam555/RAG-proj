# Traffic Law & Vehicle Regulations RAG Assistant

A RAG-powered document assistant that answers questions about traffic laws and vehicle
regulations, grounded in a set of source documents, with cited sources. **Extended Track:**
also includes a YOLO-based vision component so users can upload a vehicle/plate photo and
get answers that combine retrieved text with what was actually detected in the image.

## Overview

| | |
|---|---|
| **Domain** | Traffic laws & vehicle regulations (speed limits, plates, right of way, parking, DUI/safety, registration) |
| **Track** | Extended (RAG + YOLO vision component) |
| **Pipeline** | Chunking → Embeddings (`all-MiniLM-L6-v2`) → Chroma vector store → Retrieval → Ollama LLM generation |
| **Backend** | FastAPI (`/health`, `/query`, `/query-with-image`) |
| **Frontend** | Streamlit chat interface (text + image tabs) |
| **LLM** | Local Ollama model (default `llama3.2`) |

## Architecture

```
                    ┌─────────────────────────────┐
                    │   notebooks/rag_pipeline     │
                    │  .ipynb (offline, one-time)  │
                    │                              │
 data/documents/ ──▶│  load → chunk → embed →      │
 data/images/    ──▶│  persist Chroma vector store │
                    └──────────────┬───────────────┘
                                   │ writes to
                                   ▼
                    backend/data/vector_store/
                                   │ loaded once at startup
                                   ▼
 ┌──────────────┐   POST /query    ┌───────────────────────┐
 │  Streamlit    │ ───────────────▶│   FastAPI backend      │
 │  frontend     │◀─────────────── │  retrieval → prompt →  │
 │  (chat UI)    │  answer+sources │  Ollama LLM → answer    │
 └──────────────┘                 │  + /query-with-image:   │
                                   │  YOLO detection → fused │
                                   │  into prompt context    │
                                   └───────────────────────┘
```

## Tech Stack

- **Notebook:** Python, `pypdf`, `chromadb`, `sentence-transformers`, `pandas`
- **Backend:** FastAPI, Pydantic Settings, `ollama` (Python client), `chromadb`, `ultralytics` (YOLOv8)
- **Frontend:** Streamlit, `requests`
- **LLM runtime:** [Ollama](https://ollama.com) (runs locally, no external API calls)

## Project Structure

```
rag-assistant-project/
├── notebooks/
│   └── rag_pipeline.ipynb        # Phases 2.1–2.7: build & evaluate the pipeline
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, CORS, startup loading
│   │   ├── api/routes/query.py    # GET /health, POST /query, POST /query-with-image
│   │   ├── core/config.py         # Settings from .env
│   │   ├── schemas/query.py       # Request/response models
│   │   ├── services/
│   │   │   ├── retrieval.py       # Load vector store, retrieve chunks
│   │   │   ├── generation.py      # Call Ollama LLM, build answer
│   │   │   └── vision.py          # Extended Track: YOLO detection
│   │   └── utils/logging_config.py
│   ├── data/vector_store/         # persisted by the notebook, loaded by the backend
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py                     # Streamlit chat UI (text + image tabs)
│   ├── api_client.py              # backend API wrapper
│   ├── .env                       # API_BASE_URL=http://localhost:8000
│   └── requirements.txt
├── data/
│   ├── documents/                 # 6 source .txt documents (traffic-law domain)
│   └── images/                    # 12 sample vehicle images + YOLO-format labels.txt
├── .gitignore
└── README.md
```

## Domain & Data

Six short reference documents covering: speed limits & enforcement, license plate
requirements, right-of-way rules, parking regulations, DUI & safety equipment, and vehicle
registration/inspection. For the Extended Track vision component, `data/images/` contains a
small sample set of vehicle images with ground-truth plate bounding boxes (YOLO format) in
`labels.txt` — swap in a larger real-world dataset (e.g. a Kaggle car-plate dataset) and/or
a custom fine-tuned YOLO checkpoint for production use.

## Setup

### 0. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed, with a model pulled: `ollama pull llama3.2`
- `ollama serve` running in the background

### 1. Build the vector store (run once)
```bash
cd notebooks
pip install jupyter pandas numpy chromadb sentence-transformers pypdf ollama python-dotenv ultralytics
jupyter notebook rag_pipeline.ipynb
# Run all cells top to bottom (Kernel → Restart & Run All)
```
This persists the vector store into `backend/data/vector_store/`.

### 2. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
# open http://localhost:8000/docs to try /query from Swagger UI
```

### 3. Frontend
```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 4. Run tests
```bash
cd backend
pytest tests/ -v
```

## Environment Variables

**backend/.env**

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_PATH` | `data/vector_store` | Path to the persisted Chroma store |
| `COLLECTION_NAME` | `traffic_docs` | Chroma collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `OLLAMA_MODEL` | `llama3.2` | Local Ollama model to use for generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `TOP_K` | `3` | Number of chunks retrieved per query |
| `FRONTEND_ORIGIN` | `http://localhost:8501` | Allowed CORS origin |

**frontend/.env**

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Backend base URL |

## API Reference

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "vector_store_loaded": true, "num_chunks": 24}
```

### `POST /query`
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the speed limit near schools?"}'
```
```json
{
  "answer": "The speed limit near schools is 30 km/h during posted hours... (Source: 01_speed_limits.txt)",
  "sources": ["01_speed_limits.txt"],
  "retrieved": [{"source": "01_speed_limits.txt", "snippet": "...", "score": 0.81}]
}
```

### `POST /query-with-image` (Extended Track)
```bash
curl -X POST http://localhost:8000/query-with-image \
  -F "question=Is this plate readable?" \
  -F "image=@data/images/car_01.jpg"
```

## Evaluation Results

10 test questions were run against the retriever (see notebook Section 2.6). Retrieval
returned the correct source document as the top hit for **10/10** questions. With Ollama
running, answers were manually checked for grounding; the model correctly declined to
answer an out-of-scope question rather than hallucinating. The main observed failure mode
was very short/ambiguous queries occasionally splitting retrieval across two related
documents — mitigated by keeping `k=3` and instructing the model to say when context is
insufficient rather than blending unrelated rules. Full details and the results table are
in `notebooks/rag_pipeline.ipynb`, Section 2.6.

## Screenshots

*(Add screenshots of the running Streamlit app and Swagger `/docs` page here before
submission.)*

## Notes

- The vector store is rebuilt by re-running the notebook; it is not committed to git (see
  `.gitignore`) since it's regenerable and can grow large.
- `yolov8n.pt` (COCO-pretrained) is used by default for the vision component; swap in a
  custom-trained plate-detection checkpoint for higher accuracy on real plate images.
