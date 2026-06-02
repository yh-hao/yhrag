from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "rag_collection"
    milvus_dimension: int = 1536
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # Ollama Configuration
    ollama_model: str = "qwen3-embedding:latest"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 1
    
    # Logging
    log_level: str = "INFO"
    
    # Caching
    cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()