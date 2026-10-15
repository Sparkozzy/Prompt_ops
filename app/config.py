from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PORT: int = 8000

    # Dedicated PostgreSQL (PromptOps)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/prompt_ops_db"

    # Redis / ARQ
    REDIS_URL: str = "redis://localhost:6379/0"

    # Supabase Production
    SUPABASE_URL: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Evaluations
    EVAL_JUDGE_MODEL: str = "gpt-4o"
    EVAL_APPROVAL_THRESHOLD: float = 0.90

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
