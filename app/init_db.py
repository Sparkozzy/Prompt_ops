import asyncio
import sys
from app.database import engine
from app.models import Base

async def init_db():
    print("[DB Init] Conectando ao PostgreSQL dedicado...")
    async with engine.begin() as conn:
        print("[DB Init] Criando tabelas (se não existirem)...")
        await conn.run_sync(Base.metadata.create_all)
    print("[DB Init] Tabelas criadas com sucesso!")

if __name__ == "__main__":
    asyncio.run(init_db())
