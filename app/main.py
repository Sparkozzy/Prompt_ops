from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine
from app.models import Base
from app.routers import prompts, arena, evals, generator

app = FastAPI(
    title="PromptOps Backend — MindFlow",
    description="Plataforma interna para criação, versionamento, arena de testes e pipeline de evaluations com Quality Gate de 90%.",
    version="0.1.0"
)

# CORS liberado para o Hub frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    """Garante que as tabelas e novas colunas do PostgreSQL dedicado existam ao iniciar."""
    from sqlalchemy import text
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("ALTER TABLE prompt_versions ADD COLUMN IF NOT EXISTS channel VARCHAR(50) DEFAULT 'whatsapp';"))
        await conn.execute(text("ALTER TABLE prompt_eval_datasets ADD COLUMN IF NOT EXISTS channel VARCHAR(50) DEFAULT 'whatsapp';"))
    print("[PromptOps Startup] Tabelas e migrações do PostgreSQL dedicado concluídas.")

# Inclusão dos roteadores da API v1
app.include_router(prompts.router, prefix="/api/v1")
app.include_router(arena.router, prefix="/api/v1")
app.include_router(evals.router, prefix="/api/v1")
app.include_router(generator.router, prefix="/api/v1")

# Servidor MCP Nativo (SSE) de Contexto, RAG Parent-Child e Diagnóstico Multi-Tenant
try:
    from app.services.context_mcp_service import prompt_ops_mcp
    app.mount("/mcp", prompt_ops_mcp.sse_app())
except Exception as mcp_mount_err:
    print(f"[PromptOps MCP Mount Warning] {mcp_mount_err}")

@app.get("/health", tags=["Health"])
async def health_check():
    """Verificação de integridade do serviço."""
    return {
        "status": "healthy",
        "service": "prompt_ops",
        "environment": settings.ENVIRONMENT,
        "eval_threshold": settings.EVAL_APPROVAL_THRESHOLD
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "MindFlow PromptOps API está online.",
        "docs": "/docs"
    }
