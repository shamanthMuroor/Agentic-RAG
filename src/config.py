import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # must run before torch is imported anywhere

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    env_mode: str = "local"
    local_base_url: str = "http://localhost:1234/v1"
    local_model: str = "qwen2.5-3b-instruct"    #"qwen_qwen3.5-0.8b"
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    cloud_base_url: str = "https://api.groq.com/openai/v1"
    cloud_model: str = "openai/gpt-oss-120b"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "./chroma_db"

settings = Settings()