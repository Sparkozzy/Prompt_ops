import asyncio
import time
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db_session
from app.models import PromptVersion
from app.schemas import ArenaSingleTestRequest, ArenaCompareRequest, ArenaTurnResponse
from app.services.llm_service import generate_chat_response, render_prompt_template

router = APIRouter(prefix="/arena", tags=["Arena"])

@router.post("/test-single", response_model=ArenaTurnResponse)
async def test_single_prompt(
    payload: ArenaSingleTestRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """Executa um turno de teste contra uma única versão do prompt."""
    stmt = select(PromptVersion).where(PromptVersion.id == payload.version_id)
    res = await session.execute(stmt)
    version = res.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Versão não encontrada.")

    start_time = time.time()
    rendered_prompt = render_prompt_template(version.prompt_text, payload.variables)
    
    chat_result = await generate_chat_response(
        system_prompt=rendered_prompt,
        user_message=payload.user_message,
        chat_history=payload.chat_history,
        model=version.model_name,
        temperature=version.temperature
    )
    latency_ms = int((time.time() - start_time) * 1000)

    return ArenaTurnResponse(
        version_id=version.id,
        version_label=version.version_label,
        assistant_message=chat_result["content"],
        tool_calls=[],
        tokens_input=chat_result["tokens_input"],
        tokens_output=chat_result["tokens_output"],
        latency_ms=latency_ms
    )

@router.post("/test-compare")
async def test_compare_prompts(
    payload: ArenaCompareRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """Executa o teste concorrente lado a lado entre duas versões (A vs B)."""
    stmt_a = select(PromptVersion).where(PromptVersion.id == payload.version_a_id)
    stmt_b = select(PromptVersion).where(PromptVersion.id == payload.version_b_id)
    
    res_a, res_b = await asyncio.gather(session.execute(stmt_a), session.execute(stmt_b))
    ver_a = res_a.scalar_one_or_none()
    ver_b = res_b.scalar_one_or_none()

    if not ver_a or not ver_b:
        raise HTTPException(status_code=404, detail="Uma ou ambas versões não foram encontradas.")

    # Concorrência simultânea
    async def run_ver(ver: PromptVersion):
        start = time.time()
        rendered = render_prompt_template(ver.prompt_text, payload.variables)
        res = await generate_chat_response(
            system_prompt=rendered,
            user_message=payload.user_message,
            chat_history=payload.chat_history,
            model=ver.model_name,
            temperature=ver.temperature
        )
        latency = int((time.time() - start) * 1000)
        return ArenaTurnResponse(
            version_id=ver.id,
            version_label=ver.version_label,
            assistant_message=res["content"],
            tool_calls=[],
            tokens_input=res["tokens_input"],
            tokens_output=res["tokens_output"],
            latency_ms=latency
        )

    response_a, response_b = await asyncio.gather(run_ver(ver_a), run_ver(ver_b))

    return {
        "version_a": response_a,
        "version_b": response_b
    }
