from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db_session
from app.schemas import MetaGeneratorRequest, MetaGeneratorResponse
from app.services.llm_service import (
    DEFAULT_CORPORATE_ACTIVE_MEMORY_XML,
    generate_prompt_meta_suggestion,
)
from app.services.rag_service import (
    seed_active_and_sdr_prompts_to_rag,
    search_rag_chunks,
)

router = APIRouter(prefix="/generator", tags=["Prompt Generator & RAG"])


@router.get("/memory")
async def get_corporate_generator_active_memory():
    """
    Retorna o bloco <memoria_ativa> padrão do Gerador de Prompts Corporativos de Clientes.
    """
    return {"active_memory_xml": DEFAULT_CORPORATE_ACTIVE_MEMORY_XML}


@router.post("/suggest", response_model=MetaGeneratorResponse)
async def suggest_prompt_modifications(payload: MetaGeneratorRequest):
    """
    Agente Gerador de Prompt Corporativo de Clientes (Com Autoaprendizado - Fases 0 a 3B + Context MCP On-Demand):
    Recebe o prompt atual do cliente, a instrução e metadados de contexto (conversa da Arena A/B, prompt_id, canal),
    acionando sob demanda as tools do MCP (RAG Parent-Child, Supabase do Cliente, Arena/Evals, Schemas MCP, Docs)
    e retornando a versão refinada, auditoria Self-Refine (Fase 3B), memória atualizada, diff e tools consultadas.
    """
    res = await generate_prompt_meta_suggestion(
        current_prompt=payload.current_prompt,
        instruction=payload.instruction,
        failing_examples=payload.failing_examples,
        active_memory_xml=payload.active_memory_xml,
        prompt_id=payload.prompt_id,
        client_id=payload.client_id,
        channel=payload.channel,
        version_id=payload.version_id,
        arena_history=payload.arena_history,
        chat_history=payload.chat_history,
    )
    return MetaGeneratorResponse(
        suggested_prompt=res["suggested_prompt"],
        prompt_modified=res.get("prompt_modified", False),
        diff_unified=res["diff_unified"],
        explanation=res["explanation"],
        audit_report=res.get("audit_report"),
        updated_active_memory_xml=res.get("updated_active_memory_xml"),
        tools_consulted=res.get("tools_consulted", []),
    )


@router.post("/rag/seed")
async def seed_rag_parent_child_prompts(
    session: AsyncSession = Depends(get_db_session),
):
    """
    Indexa no RAG Parent-Child (PostgreSQL dedicado) apenas os prompts vinculados a clientes ativos
    (client_configurations) e os prompts de Treinamento de SDR (IDs 32 e 67).
    """
    return await seed_active_and_sdr_prompts_to_rag(session)


@router.get("/rag/search")
async def search_rag_parent_child_blocks(
    query: str = Query(..., description="Consulta semântica para buscar blocos de prompts"),
    section_tag: Optional[str] = Query(None, description="Filtro opcional de tag (ex: few_shot_declarado)"),
    channel: Optional[str] = Query(None, description="Filtro opcional de canal (whatsapp, ligacao, fup)"),
    exclude_prompt_id: Optional[int] = Query(None),
    include_parent_text: bool = Query(False),
    top_k: int = Query(4),
    session: AsyncSession = Depends(get_db_session),
):
    """Busca semântica híbrida nos Child Chunks do RAG Parent-Child."""
    chunks = await search_rag_chunks(
        session=session,
        query=query,
        section_tag=section_tag,
        channel=channel,
        exclude_prompt_id=exclude_prompt_id,
        include_parent_text=include_parent_text,
        top_k=top_k,
    )
    return {
        "query": query,
        "total_results": len(chunks),
        "chunks": chunks,
    }
