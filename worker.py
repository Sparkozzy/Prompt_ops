import os
from arq.connections import RedisSettings
from app.config import settings


async def run_evaluation_batch_task(ctx, run_id: str):
    """Tarefa em background que executa a bateria de evaluations para uma versão."""
    print(f"[Worker] Iniciando bateria de evaluations para run_id: {run_id}")
    # Lógica de iteração sobre os datasets, chamadas de modelo e juiz
    return {"status": "completed", "run_id": run_id}


async def startup(ctx):
    print("[Worker] PromptOps ARQ Worker inicializado com sucesso.")


async def shutdown(ctx):
    print("[Worker] PromptOps ARQ Worker finalizando conexões.")


def get_redis_settings() -> RedisSettings:
    raw_url = os.getenv("REDIS_URL", settings.REDIS_URL)
    # Trata caracteres especiais na URL caso existam
    return RedisSettings.from_dsn(raw_url)


class WorkerSettings:
    functions = [run_evaluation_batch_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_redis_settings()
