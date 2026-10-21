from fastapi import APIRouter
from app.schemas import MetaGeneratorRequest, MetaGeneratorResponse
from app.services.llm_service import (
    DEFAULT_CORPORATE_ACTIVE_MEMORY_XML,
    generate_prompt_meta_suggestion,
)

router = APIRouter(prefix="/generator", tags=["Prompt Generator"])


@router.get("/memory")
async def get_corporate_generator_active_memory():
    """
    Retorna o bloco <memoria_ativa> padrão do Gerador de Prompts Corporativos de Clientes.
    """
    return {"active_memory_xml": DEFAULT_CORPORATE_ACTIVE_MEMORY_XML}


@router.post("/suggest", response_model=MetaGeneratorResponse)
async def suggest_prompt_modifications(payload: MetaGeneratorRequest):
    """
    Agente Gerador de Prompt Corporativo de Clientes (Com Autoaprendizado - Fases 0 a 3B):
    Recebe o prompt atual do cliente (tabela Prompts), as instruções/briefing e a memória ativa,
    gerando a versão refinada, auditoria Self-Refine (Fase 3B), memória atualizada e o diff unificado.
    """
    res = await generate_prompt_meta_suggestion(
        current_prompt=payload.current_prompt,
        instruction=payload.instruction,
        failing_examples=payload.failing_examples,
        active_memory_xml=payload.active_memory_xml,
    )
    return MetaGeneratorResponse(
        suggested_prompt=res["suggested_prompt"],
        diff_unified=res["diff_unified"],
        explanation=res["explanation"],
        audit_report=res.get("audit_report"),
        updated_active_memory_xml=res.get("updated_active_memory_xml"),
    )
