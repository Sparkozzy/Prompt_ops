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
    """Executa um turno de teste contra uma única versão do prompt com tools reais."""
    stmt = select(PromptVersion).where(PromptVersion.id == payload.version_id)
    res = await session.execute(stmt)
    version = res.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Versão não encontrada.")

    from app.services.mcp_service import get_client_mcp_urls, get_client_tools_and_map
    mcp_urls = await get_client_mcp_urls(prompt_id=version.prompt_id, client_id=version.client_id)
    tools, tool_map = await get_client_tools_and_map(mcp_urls)

    start_time = time.time()
    rendered_prompt = render_prompt_template(version.prompt_text, payload.variables)
    
    chat_result = await generate_chat_response(
        system_prompt=rendered_prompt,
        user_message=payload.user_message,
        chat_history=payload.chat_history,
        model=version.model_name,
        temperature=version.temperature,
        tools=tools,
        mcp_tool_map=tool_map,
        version_id=version.id
    )
    latency_ms = int((time.time() - start_time) * 1000)

    # Converte os tool_calls para o schema ToolCallDetail
    from app.schemas import ToolCallDetail
    formatted_tools = [
        ToolCallDetail(
            tool_name=t["tool_name"],
            input_payload=t["input"],
            output_payload={"result": t["output"]},
            latency_ms=t.get("latency_ms", 0),
            status=t.get("status", "success")
        )
        for t in chat_result.get("tool_calls_executed", [])
    ]

    return ArenaTurnResponse(
        version_id=version.id,
        version_label=version.version_label,
        assistant_message=chat_result["content"],
        tool_calls=formatted_tools,
        tokens_input=chat_result["tokens_input"],
        tokens_output=chat_result["tokens_output"],
        latency_ms=latency_ms
    )

@router.post("/test-compare")
async def test_compare_prompts(
    payload: ArenaCompareRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """Executa o teste concorrente lado a lado entre duas versões (A vs B) com tools reais."""
    stmt_a = select(PromptVersion).where(PromptVersion.id == payload.version_a_id)
    stmt_b = select(PromptVersion).where(PromptVersion.id == payload.version_b_id)
    
    res_a = await session.execute(stmt_a)
    res_b = await session.execute(stmt_b)
    ver_a = res_a.scalar_one_or_none()
    ver_b = res_b.scalar_one_or_none()

    if not ver_a or not ver_b:
        raise HTTPException(status_code=404, detail="Uma ou ambas versões não foram encontradas.")

    from app.services.mcp_service import get_client_mcp_urls, get_client_tools_and_map
    from app.schemas import ToolCallDetail

    mcp_urls_a = await get_client_mcp_urls(prompt_id=ver_a.prompt_id, client_id=ver_a.client_id)
    tools_a, map_a = await get_client_tools_and_map(mcp_urls_a)

    mcp_urls_b = await get_client_mcp_urls(prompt_id=ver_b.prompt_id, client_id=ver_b.client_id)
    tools_b, map_b = await get_client_tools_and_map(mcp_urls_b)

    async def run_ver(ver: PromptVersion, tools: list, t_map: dict):
        start = time.time()
        rendered = render_prompt_template(ver.prompt_text, payload.variables)
        res = await generate_chat_response(
            system_prompt=rendered,
            user_message=payload.user_message,
            chat_history=payload.chat_history,
            model=ver.model_name,
            temperature=ver.temperature,
            tools=tools,
            mcp_tool_map=t_map,
            version_id=ver.id
        )
        latency = int((time.time() - start) * 1000)
        formatted_tools = [
            ToolCallDetail(
                tool_name=t["tool_name"],
                input_payload=t["input"],
                output_payload={"result": t["output"]},
                latency_ms=t.get("latency_ms", 0),
                status=t.get("status", "success")
            )
            for t in res.get("tool_calls_executed", [])
        ]
        return ArenaTurnResponse(
            version_id=ver.id,
            version_label=ver.version_label,
            assistant_message=res["content"],
            tool_calls=formatted_tools,
            tokens_input=res["tokens_input"],
            tokens_output=res["tokens_output"],
            latency_ms=latency
        )

    import traceback
    try:
        # As duas chamadas ao LLM continuam simultâneas (não usam a sessão do banco).
        response_a, response_b = await asyncio.gather(run_ver(ver_a, tools_a, map_a), run_ver(ver_b, tools_b, map_b))
    except Exception as e:
        print("[ARENA TEST-COMPARE ERROR]", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Falha na comparação da Arena: {str(e)}")

    return {
        "version_a": response_a,
        "version_b": response_b
    }
