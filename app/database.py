from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from supabase import create_client, Client
from app.config import settings

# Engine assíncrono para o banco dedicado do PromptOps
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency para injeção de sessão do banco dedicado em rotas FastAPI."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# Cliente Singleton Supabase (Produção)
_supabase_client: Optional[Client] = None


def get_production_supabase() -> Client:
    """Retorna o cliente singleton do Supabase Produção.
    Usado ESTRITAMENTE para ler prompt atual ou publicar versão aprovada.
    """
    global _supabase_client
    if _supabase_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY são obrigatórios.")
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
    return _supabase_client
