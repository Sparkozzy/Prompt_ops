from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

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
