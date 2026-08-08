from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Vector store
    vector_store_path: str = "data/vector_store"
    collection_name: str = "traffic_docs"
    embedding_model: str = "all-MiniLM-L6-v2"

    # LLM
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    # Retrieval
    top_k: int = 3

    # CORS
    frontend_origin: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
