from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db_session, get_production_supabase
from app.models import PromptVersion, get_utc_now
from app.schemas import (
    PromptVersionCreate,
    PromptVersionRead,
    PromptPublishRequest,
    PromptPublishResponse,
    PromptRollbackRequest,
    PromptCreateRequest,
    PromptCreateResponse,
)

router = APIRouter(prefix="/prompts", tags=["Prompts"])

@router.get("/production/list")
async def list_production_prompts():
    """Lista todos os prompts oficiais diretamente da tabela Prompts no Supabase com mapeamento de canais."""
    supabase = get_production_supabase()
    res = supabase.table("Prompts").select(
        'id, created_at, "Nome do cliente", "Pormpt_Name", "Prompt_Text", "Ligação/txt", "prompt_fup", "Prompt Insta"'
    ).order("id", desc=True).limit(100).execute()

    items = []
    for p in res.data:
        channels = []
        if p.get("Prompt_Text") and str(p.get("Prompt_Text")).strip():
            channels.append("whatsapp")
        if p.get("Ligação/txt") and str(p.get("Ligação/txt")).strip():
            channels.append("ligacao")
        if p.get("prompt_fup") and str(p.get("prompt_fup")).strip():
            channels.append("fup")
        if p.get("Prompt Insta") and str(p.get("Prompt Insta")).strip():
            channels.append("instagram")
        if not channels:
            channels.append("whatsapp")

        items.append({
            "id": p["id"],
            "created_at": p.get("created_at"),
            "Nome do cliente": p.get("Nome do cliente") or "Sem cliente",
            "Pormpt_Name": p.get("Pormpt_Name") or "Prompt",
            "channels": channels,
            "has_whatsapp": "whatsapp" in channels,
            "has_ligacao": "ligacao" in channels,
            "has_fup": "fup" in channels,
            "has_insta": "instagram" in channels
        })
    return items

@router.post("/production", status_code=201)
async def create_production_prompt(
    payload: PromptCreateRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Cria uma nova linha de prompt no Supabase oficial e registra a versão inicial no banco dedicado.
    Mapeia o canal para a coluna correta (WhatsApp -> Prompt_Text, Ligação -> Ligação/txt, FUP -> prompt_fup, Insta -> Prompt Insta).
    """
    client_name = (payload.client_name or "").strip()
    if not client_name:
        raise HTTPException(status_code=400, detail="Nome do cliente é obrigatório.")

    channel = (payload.channel or "whatsapp").lower().strip()
    col_mapping = {
        "whatsapp": "Prompt_Text",
        "ligacao": "Ligação/txt",
        "fup": "prompt_fup",
        "instagram": "Prompt Insta"
    }
    target_col = col_mapping.get(channel, "Prompt_Text")

    prompt_body = payload.prompt_text or ""
    row_data = {
        "Nome do cliente": client_name,
        "Pormpt_Name": payload.prompt_name or "atendimento_geral",
        "Prompt_Text": prompt_body if target_col == "Prompt_Text" else None,
        "Ligação/txt": prompt_body if target_col == "Ligação/txt" else None,
        "prompt_fup": prompt_body if target_col == "prompt_fup" else None,
        "Prompt Insta": prompt_body if target_col == "Prompt Insta" else None,
    }

    supabase = get_production_supabase()

    # Obtém o próximo ID para evitar colisão caso a sequence do PostgreSQL esteja desincronizada
    try:
        max_res = supabase.table("Prompts").select("id").order("id", desc=True).limit(1).execute()
        next_id = (max_res.data[0]["id"] if max_res.data else 0) + 1
        row_data["id"] = next_id
    except Exception:
        pass

    try:
        res = supabase.table("Prompts").insert(row_data).execute()
    except Exception as e:
        # Fallback sem ID explícito caso o banco recuse inserção com chave primária explícita
        if "id" in row_data:
            del row_data["id"]
            res = supabase.table("Prompts").insert(row_data).execute()
        else:
            raise e

    if not res.data:
        raise HTTPException(status_code=500, detail="Erro ao inserir prompt no Supabase.")

    created = res.data[0]
    p_id = created["id"]

    # Se client_id fornecido, associa na tabela client_configurations
    if payload.client_id:
        try:
            supabase.table("client_configurations").update({
                "prompt_id": p_id
            }).eq("client_id", str(payload.client_id)).execute()
        except Exception:
            pass

    # Cria snapshot de versão inicial no banco de dados dedicado do PromptOps
    try:
        initial_version = PromptVersion(
            prompt_id=p_id,
            client_id=str(payload.client_id or p_id),
            client_name=client_name,
            channel=channel,
            version_label=f"prod-{channel}-1",
            prompt_text=prompt_body,
            model_name="gpt-4.1",
            temperature=0.7,
            commit_message="Versão inicial de criação",
            author_email=payload.author_email or "admin@mindflow.ai",
            status="published",
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
            published_at=get_utc_now()
        )
        session.add(initial_version)
        await session.commit()
    except Exception:
        pass

    # Sincroniza RAG se houver texto inicial
    if prompt_body.strip():
        try:
            from app.services.rag_service import upsert_prompt_in_rag
            await upsert_prompt_in_rag(
                db=session,
                prompt_id=p_id,
                client_id=str(payload.client_id or p_id),
                client_name=client_name,
                channel=channel,
                prompt_text=prompt_body
            )
        except Exception:
            pass

    channels = [channel]
    return {
        "id": p_id,
        "created_at": created.get("created_at"),
        "Nome do cliente": created.get("Nome do cliente"),
        "Pormpt_Name": created.get("Pormpt_Name"),
        "channel": channel,
        "Prompt_Text": prompt_body,
        "channels": channels
    }

@router.get("/production/{prompt_id}")
async def get_production_prompt(prompt_id: int, channel: str = Query("whatsapp")):
    """Busca um prompt oficial específico no Supabase para importação/visualização conforme o canal."""
    supabase = get_production_supabase()
    res = supabase.table("Prompts").select("*").eq("id", prompt_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Prompt não encontrado no Supabase.")

    data = res.data
    # Extrai o texto conforme o canal solicitado
    ch = (channel or "whatsapp").lower()
    if ch in ["ligacao", "ligação", "voz"]:
        text = data.get("Ligação/txt") or ""
        resolved_channel = "ligacao"
    elif ch in ["fup", "followup", "follow_up"]:
        text = data.get("prompt_fup") or ""
        resolved_channel = "fup"
    elif ch in ["insta", "instagram"]:
        text = data.get("Prompt Insta") or ""
        resolved_channel = "instagram"
    else:
        text = data.get("Prompt_Text") or ""
        resolved_channel = "whatsapp"

    data["Prompt_Text"] = text
    data["channel"] = resolved_channel
    return data

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
        channel=payload.channel or "whatsapp",
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
    channel: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db_session)
):
    """Lista o histórico de versões do banco dedicado, com filtro opcional por canal."""
    stmt = select(PromptVersion).order_by(desc(PromptVersion.created_at))
    if prompt_id is not None:
        stmt = stmt.where(PromptVersion.prompt_id == prompt_id)
    if client_id is not None:
        stmt = stmt.where(PromptVersion.client_id == client_id)
    if channel is not None:
        stmt = stmt.where(PromptVersion.channel == channel)
    
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
    Atualiza a linha existente na coluna correta da tabela Prompts do Supabase mantendo o mesmo id.
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

    # Identifica a coluna oficial correspondente ao canal
    ch = (getattr(version, "channel", "whatsapp") or "whatsapp").lower()
    target_column = "Prompt_Text"
    if ch in ["ligacao", "ligação", "voz"]:
        target_column = "Ligação/txt"
    elif ch in ["fup", "followup", "follow_up"]:
        target_column = "prompt_fup"
    elif ch in ["insta", "instagram"]:
        target_column = "Prompt Insta"

    # Atualiza no Supabase oficial (Prompts)
    supabase = get_production_supabase()
    update_res = supabase.table("Prompts").update({
        target_column: version.prompt_text
    }).eq("id", version.prompt_id).execute()

    if not update_res.data:
        raise HTTPException(status_code=500, detail=f"Falha ao atualizar coluna '{target_column}' na tabela Prompts do Supabase.")

    # Atualiza modelo e temperatura na tabela client_configurations do cliente correspondente
    try:
        config_update = {}
        if version.model_name:
            config_update["llm_model"] = version.model_name
        if version.temperature is not None:
            config_update["llm_temperature"] = version.temperature
        
        if config_update:
            # Tenta atualizar pelo prompt_id ou pelo client_id
            if version.prompt_id:
                supabase.table("client_configurations").update(config_update).eq("prompt_id", int(version.prompt_id)).execute()
            elif version.client_id:
                supabase.table("client_configurations").update(config_update).eq("client_id", str(version.client_id)).execute()
    except Exception as conf_err:
        print(f"[Publish Warning] Falha ao sincronizar client_configurations: {conf_err}")

    # Marca como publicada no banco dedicado
    version.status = "published"
    version.published_at = get_utc_now()
    await session.commit()

    # Sincroniza atomicamente o RAG Parent-Child (substitui Parent anterior e expurga Children obsoletos via CASCADE)
    try:
        from app.services.rag_service import upsert_prompt_in_rag
        await upsert_prompt_in_rag(
            session=session,
            prompt_id=int(version.prompt_id),
            client_id=str(version.client_id),
            client_name=version.client_name,
            prompt_name=version.client_name,
            channel=ch,
            prompt_text=version.prompt_text,
            version_label=version.version_label,
            eval_score=version.eval_score,
        )
    except Exception as rag_err:
        print(f"[Publish RAG Sync Warning] Falha ao atualizar RAG Parent-Child: {rag_err}")

    return PromptPublishResponse(
        success=True,
        message="Versão aprovada e publicada em produção no Supabase com sucesso!",
        prompt_id=version.prompt_id,
        version_label=version.version_label,
        approval_rate=version.eval_score,
        published_at=version.published_at
    )
