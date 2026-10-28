from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
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
from app.services.context_mcp_service import (
    read_mindflow_engineering_docs_impl,
    update_mindflow_engineering_doc_impl,
    execute_generator_tool_call,
)

router = APIRouter(prefix="/generator", tags=["Prompt Generator & RAG"])


class EngineeringDocUpdatePayload(BaseModel):
    content: str
    updated_by: Optional[str] = "antigravity_agent"
    change_summary: Optional[str] = "Atualização sincronizada"
    title: Optional[str] = None


class McpToolCallPayload(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}
    context: Dict[str, Any] = {}


@router.get("/memory")
async def get_corporate_generator_active_memory():
    """
    Retorna o bloco <memoria_ativa> sincronizado entre Antigravity e PromptOps.
    """
    doc = await read_mindflow_engineering_docs_impl(topic="active_memory_xml")
    return {
        "active_memory_xml": doc.get("content") or DEFAULT_CORPORATE_ACTIVE_MEMORY_XML,
        "version": doc.get("version", 1),
        "updated_by": doc.get("updated_by", "system"),
    }


@router.get("/docs")
async def get_shared_engineering_docs(
    doc_key: str = Query("all", description="Chave do documento (prompt_hypotheses_ledger, active_memory_xml, canonical_prompt_anatomy, supabase_data_guide ou all)"),
):
    """
    Retorna a versão viva e sincronizada dos documentos de Engenharia e Hipóteses compartilhados entre Antigravity e PromptOps.
    """
    return await read_mindflow_engineering_docs_impl(topic=doc_key)


@router.put("/docs/{doc_key}")
async def update_shared_engineering_doc(doc_key: str, payload: EngineeringDocUpdatePayload):
    """
    Atualiza de forma transacional e versionada um documento compartilhado (ex: prompt_hypotheses_ledger ou active_memory_xml).
    Qualquer alteração feita pelo Agente do Antigravity ou pelo Agente do PromptOps fica imediatamente visível para o outro.
    """
    return await update_mindflow_engineering_doc_impl(
        doc_key=doc_key,
        content=payload.content,
        updated_by=payload.updated_by or "antigravity_agent",
        change_summary=payload.change_summary or "Atualização sincronizada",
        title=payload.title,
    )


@router.post("/mcp-tool")
async def call_shared_mcp_tool(payload: McpToolCallPayload):
    """
    Endpoint unificado para execução direta das tools do prompt_ops_context_mcp
    (usado pela ponte MCP Stdio local do Antigravity e testes de integração).
    """
    return await execute_generator_tool_call(
        tool_name=payload.tool_name,
        arguments=payload.arguments,
        context=payload.context,
    )


@router.post("/suggest", response_model=MetaGeneratorResponse)
async def suggest_prompt_modifications(payload: MetaGeneratorRequest):
    """
    Agente Gerador de Prompt Corporativo de Clientes (Com Autoaprendizado - Fases 0 a 3B + Context MCP On-Demand):
    Recebe o prompt atual do cliente, a instrução e metadados de contexto (conversa da Arena A/B, prompt_id, canal),
    acionando sob demanda as tools do MCP (RAG Parent-Child, Supabase do Cliente, Arena/Evals, Schemas MCP, Docs)
    e retornando a versão refinada, auditoria Self-Refine (Fase 3B), memória atualizada, diff e tools consultadas.
    """
    live_memory = payload.active_memory_xml
    if not live_memory:
        try:
            mem_doc = await read_mindflow_engineering_docs_impl(topic="active_memory_xml")
            live_memory = mem_doc.get("content") or DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
        except Exception:
            live_memory = DEFAULT_CORPORATE_ACTIVE_MEMORY_XML

    res = await generate_prompt_meta_suggestion(
        current_prompt=payload.current_prompt,
        instruction=payload.instruction,
        failing_examples=payload.failing_examples,
        active_memory_xml=live_memory,
        prompt_id=payload.prompt_id,
        client_id=payload.client_id,
        channel=payload.channel,
        version_id=payload.version_id,
        arena_history=payload.arena_history,
        chat_history=payload.chat_history,
        attachments=[att.dict() if hasattr(att, "dict") else att.model_dump() for att in (payload.attachments or [])],
    )

    # Se o compilador do Agente do PromptOps atualizou a memória ativa durante a geração, persiste no banco compartilhado
    updated_mem = res.get("updated_active_memory_xml")
    if updated_mem and updated_mem.strip() and updated_mem.strip() != (live_memory or "").strip():
        try:
            await update_mindflow_engineering_doc_impl(
                doc_key="active_memory_xml",
                content=updated_mem,
                updated_by="prompt_ops_copilot",
                change_summary=f"Autoaprendizado Fase 0/3B: {payload.instruction[:80]}",
            )
        except Exception as e:
            print(f"[Generator Memory Sync Warning] {e}")

    return MetaGeneratorResponse(
        suggested_prompt=res["suggested_prompt"],
        prompt_modified=res.get("prompt_modified", False),
        diff_unified=res["diff_unified"],
        explanation=res["explanation"],
        audit_report=res.get("audit_report"),
        updated_active_memory_xml=updated_mem,
        tools_consulted=res.get("tools_consulted", []),
        action_proposals=res.get("action_proposals", []),
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

