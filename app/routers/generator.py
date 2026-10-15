from fastapi import APIRouter
from app.schemas import MetaGeneratorRequest, MetaGeneratorResponse
from app.services.llm_service import generate_prompt_meta_suggestion

router = APIRouter(prefix="/generator", tags=["Prompt Generator"])

@router.post("/suggest", response_model=MetaGeneratorResponse)
async def suggest_prompt_modifications(payload: MetaGeneratorRequest):
    """
    Agente Gerador de Prompt:
    Recebe o prompt atual e as instruções do engenheiro, gerando a versão refinada
    e o diff unificado (estilo git diff) para revisão antes de salvar.
    """
    res = await generate_prompt_meta_suggestion(
        current_prompt=payload.current_prompt,
        instruction=payload.instruction,
        failing_examples=payload.failing_examples
    )
    return MetaGeneratorResponse(
        suggested_prompt=res["suggested_prompt"],
        diff_unified=res["diff_unified"],
        explanation=res["explanation"]
    )
