from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db_session, get_production_supabase
from app.models import PromptVersion, get_utc_now
from app.schemas import PromptVersionCreate, PromptVersionRead, PromptPublishRequest, PromptPublishResponse, PromptRollbackRequest

router = APIRouter(prefix="/prompts", tags=["Prompts"])

@router.get("/production/list")
async def list_production_prompts():
    """Lista todos os prompts oficiais diretamente da tabela Prompts no Supabase."""
    supabase = get_production_supabase()
    res = supabase.table("Prompts").select("id, created_at, \"Nome do cliente\", \"Pormpt_Name\", \"Ligação/txt\"").order("id", desc=True).limit(50).execute()
    return res.data

@router.get("/production/{prompt_id}")
async def get_production_prompt(prompt_id: int):
    """Busca um prompt oficial específico no Supabase para importação/visualização."""
    supabase = get_production_supabase()
    res = supabase.table("Prompts").select("*").eq("id", prompt_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Prompt não encontrado no Supabase.")
    return res.data

@router.post("/versions", response_model=PromptVersionRead)
async def create_prompt_version(
    payload: PromptVersionCreate,
    session: AsyncSession = Depends(get_db_session)
):
    """Cria uma nova versão de trabalho (draft) no banco dedicado."""
    version = PromptVersion(
        prompt_id=payload.prompt_id,
        client_id=payload.client_id,
        client_name=payload.client_name,
        version_label=payload.version_label,
        prompt_text=payload.prompt_text,
        model_name=payload.model_name,
        temperature=payload.temperature,
        commit_message=payload.commit_message,
        author_email=payload.author_email,
        status="draft",
        created_at=get_utc_now(),
        updated_at=get_utc_now()
    )
    session.add(version)
    await session.commit()
    await session.refresh(version)
    return version

@router.get("/versions", response_model=List[PromptVersionRead])
async def list_prompt_versions(
    prompt_id: Optional[int] = Query(None),
    client_id: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db_session)
):
    """Lista o histórico de versões do banco dedicado."""
    stmt = select(PromptVersion).order_by(desc(PromptVersion.created_at))
    if prompt_id is not None:
        stmt = stmt.where(PromptVersion.prompt_id == prompt_id)
    if client_id is not None:
        stmt = stmt.where(PromptVersion.client_id == client_id)
    
    result = await session.execute(stmt)
    return result.scalars().all()

@router.get("/versions/{version_id}", response_model=PromptVersionRead)
async def get_prompt_version(
    version_id: UUID,
    session: AsyncSession = Depends(get_db_session)
):
    """Obtém os detalhes de uma versão específica."""
    stmt = select(PromptVersion).where(PromptVersion.id == version_id)
    result = await session.execute(stmt)
    version = result.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Versão não encontrada.")
    return version

@router.post("/publish", response_model=PromptPublishResponse)
async def publish_version_to_production(
    payload: PromptPublishRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Quality Gate & Deploy:
    Apenas publica se eval_score >= 0.90 (ou 90%).
    Atualiza a linha existente na tabela Prompts do Supabase mantendo o mesmo id.
    """
    stmt = select(PromptVersion).where(PromptVersion.id == payload.version_id)
    result = await session.execute(stmt)
    version = result.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Versão não encontrada.")

    # Quality Gate
    if version.eval_score is None or version.eval_score < 0.90:
        raise HTTPException(
            status_code=400,
            detail=f"Quality Gate Bloqueado: A versão precisa atingir no mínimo 90% de aprovação (score atual: {((version.eval_score or 0) * 100):.1f}%)."
        )

    # Atualiza no Supabase oficial
    supabase = get_production_supabase()
    now_iso = get_utc_now().isoformat()
    
    update_res = supabase.table("Prompts").update({
        "Prompt_Text": version.prompt_text
    }).eq("id", version.prompt_id).execute()

    if not update_res.data:
        raise HTTPException(status_code=500, detail="Falha ao atualizar tabela Prompts no Supabase.")

    # Marca como publicada no banco dedicado
    version.status = "published"
    version.published_at = get_utc_now()
    await session.commit()

    return PromptPublishResponse(
        success=True,
        message="Versão aprovada e publicada em produção no Supabase com sucesso!",
        prompt_id=version.prompt_id,
        version_label=version.version_label,
        approval_rate=version.eval_score,
        published_at=version.published_at
    )
