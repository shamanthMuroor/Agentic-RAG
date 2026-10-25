from langchain_openai import ChatOpenAI
from src.config import settings


def _cloud_key() -> str:
    if "openrouter" in settings.cloud_base_url:
        return settings.openrouter_api_key
    return settings.groq_api_key


def get_llm(temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    """Return a chat model for the current ENV_MODE (local | cloud)."""
    if settings.env_mode == "local":
        return ChatOpenAI(
            base_url=settings.local_base_url,
            api_key="lm-studio",
            model=settings.local_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            max_retries=1,
        )
    if settings.env_mode == "cloud":
        key = _cloud_key()
        if not key:
            raise RuntimeError("ENV_MODE=cloud but no API key set in .env")
        return ChatOpenAI(
            base_url=settings.cloud_base_url,
            api_key=key,
            model=settings.cloud_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=2,
        )
    raise ValueError(f"Unknown ENV_MODE: {settings.env_mode}")