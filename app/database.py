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
        yield session


# Cliente Singleton Supabase (Produção)
_supabase_client: Optional[Client] = None


def get_production_supabase() -> Client:
    """Retorna o cliente singleton do Supabase Produção.
    Usado ESTRITAMENTE para ler prompt atual ou publicar versão aprovada.
    """
    global _supabase_client
    if _supabase_client is None:
        url = settings.get_effective_supabase_url()
        key = settings.get_effective_supabase_key()
        if not url or not key:
            raise ValueError("SUPABASE_URL/MASTER_SUPABASE_URL e chave são obrigatórios.")
        _supabase_client = create_client(url, key)
    return _supabase_client
