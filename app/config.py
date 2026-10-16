from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PORT: int = 8000

    # Dedicated PostgreSQL (PromptOps)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/prompt_ops_db"

    # Redis / ARQ
    REDIS_URL: str = "redis://localhost:6379/0"

    # Supabase Production
    SUPABASE_URL: Optional[str] = Field(default=None, validation_alias="SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(default=None, validation_alias="SUPABASE_SERVICE_ROLE_KEY")
    MASTER_SUPABASE_URL: Optional[str] = Field(default=None, validation_alias="MASTER_SUPABASE_URL")
    MASTER_SUPABASE_SERVICE_KEY: Optional[str] = Field(default=None, validation_alias="MASTER_SUPABASE_SERVICE_KEY")
    SUPABASE_KEY: Optional[str] = Field(default=None, validation_alias="SUPABASE_KEY")

    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    Openai_api_key: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Evaluations
    EVAL_JUDGE_MODEL: str = "gpt-4o"
    EVAL_APPROVAL_THRESHOLD: float = 0.90

    def get_effective_supabase_url(self) -> str:
        return self.SUPABASE_URL or self.MASTER_SUPABASE_URL or "https://ghayhpwthdbmnpsptcnb.supabase.co"

    def get_effective_supabase_key(self) -> str:
        return self.SUPABASE_SERVICE_ROLE_KEY or self.MASTER_SUPABASE_SERVICE_KEY or self.SUPABASE_KEY or ""

    def get_effective_openai_key(self) -> str:
        return self.OPENAI_API_KEY or self.Openai_api_key or ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
