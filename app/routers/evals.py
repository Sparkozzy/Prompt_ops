import asyncio
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db_session
from app.models import EvalDataset, EvalRun, EvalCaseResult, PromptVersion, get_utc_now
from app.schemas import EvalDatasetCreate, EvalDatasetUpdate, EvalDatasetRead, EvalRunTrigger, EvalRunDetailRead
from app.services.eval_service import evaluate_single_turn_case

router = APIRouter(prefix="/evals", tags=["Evaluations"])

@router.post("/datasets", response_model=EvalDatasetRead)
async def create_eval_dataset_case(
    payload: EvalDatasetCreate,
    session: AsyncSession = Depends(get_db_session)
):
    """Cria um caso de teste no dataset dedicado ao cliente/prompt."""
    item = EvalDataset(
        prompt_id=payload.prompt_id,
        client_id=payload.client_id,
        channel=payload.channel or "whatsapp",
        scenario_name=payload.scenario_name,
        test_type=payload.test_type,
        input_data=payload.input_data,
        expected_criteria=payload.expected_criteria,
        requires_tool=payload.requires_tool,
        expected_tool_name=payload.expected_tool_name,
        is_active=True,
        created_at=get_utc_now()
    )
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return item

@router.get("/datasets", response_model=List[EvalDatasetRead])
async def list_eval_datasets(
    prompt_id: Optional[int] = Query(None),
    client_id: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db_session)
):
    """Lista todos os casos de teste ativos para um prompt/cliente."""
    stmt = select(EvalDataset).where(EvalDataset.is_active == True).order_by(desc(EvalDataset.created_at))
    if prompt_id is not None:
        stmt = stmt.where(EvalDataset.prompt_id == prompt_id)
    if client_id is not None:
        stmt = stmt.where(EvalDataset.client_id == client_id)
    if channel is not None:
        stmt = stmt.where(EvalDataset.channel == channel)
    res = await session.execute(stmt)
    return res.scalars().all()

@router.post("/run", response_model=EvalRunDetailRead)
async def run_evaluations_batch(
    payload: EvalRunTrigger,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Executa a bateria de avaliações completa para uma versão de prompt.
    Calcula aprovação percentual e atualiza o eval_score na versão.
    """
    stmt_v = select(PromptVersion).where(PromptVersion.id == payload.version_id)
    res_v = await session.execute(stmt_v)
    version = res_v.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Versão não encontrada.")

    # Busca casos de teste filtrando por prompt_id e opcionalmente canal
    stmt_d = select(EvalDataset).where(
        EvalDataset.prompt_id == version.prompt_id,
        EvalDataset.is_active == True
    )
    if hasattr(version, "channel") and version.channel:
        stmt_d = stmt_d.where(EvalDataset.channel == version.channel)

    res_d = await session.execute(stmt_d)
    cases = res_d.scalars().all()

    # Fallback se não houver casos filtrados por canal específico
    if not cases:
        stmt_fallback = select(EvalDataset).where(
            EvalDataset.prompt_id == version.prompt_id,
            EvalDataset.is_active == True
        )
        res_fb = await session.execute(stmt_fallback)
        cases = res_fb.scalars().all()

    if not cases:
        raise HTTPException(status_code=400, detail="Nenhum caso de teste ativo cadastrado para este prompt.")

    run = EvalRun(
        version_id=version.id,
        prompt_id=version.prompt_id,
        status="running",
        total_cases=len(cases),
        passed_cases=0,
        approval_rate=0.0,
        triggered_by=payload.triggered_by,
        started_at=get_utc_now()
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    passed_count = 0
    case_results = []

    for c in cases:
        eval_result = await evaluate_single_turn_case(
            prompt_text=version.prompt_text,
            model_name=version.model_name,
            temperature=version.temperature,
            case_input=c.input_data,
            expected_criteria=c.expected_criteria,
            prompt_id=version.prompt_id,
            client_id=version.client_id,
            requires_tool=c.requires_tool,
            expected_tool_name=c.expected_tool_name,
            version_id=version.id
        )
        if eval_result["passed"]:
            passed_count += 1

        case_item = EvalCaseResult(
            run_id=run.id,
            dataset_id=c.id,
            passed=eval_result["passed"],
            generated_response=eval_result["generated_response"],
            judge_reasoning=eval_result["judge_reasoning"],
            tool_called=eval_result.get("tool_called"),
            tool_call_valid=eval_result.get("tool_call_valid"),
            execution_time_ms=eval_result["execution_time_ms"]
        )
        session.add(case_item)
        case_results.append(case_item)

    approval_rate = passed_count / len(cases)
    run.passed_cases = passed_count
    run.approval_rate = approval_rate
    run.status = "completed"
    run.completed_at = get_utc_now()

    # Atualiza eval_score na versão
    version.eval_score = approval_rate
    if approval_rate >= 0.90:
        version.status = "approved"

    await session.commit()
    
    # Recarrega o run com os cases relacionados via selectinload
    from sqlalchemy.orm import selectinload
    stmt_run = select(EvalRun).options(selectinload(EvalRun.cases)).where(EvalRun.id == run.id)
    result_run = await session.execute(stmt_run)
    return result_run.scalar_one()


@router.put("/datasets/{dataset_id}", response_model=EvalDatasetRead)
async def update_eval_dataset_case(
    dataset_id: UUID,
    payload: EvalDatasetUpdate,
    session: AsyncSession = Depends(get_db_session)
):
    """Atualiza os dados de um caso de teste existente."""
    stmt = select(EvalDataset).where(EvalDataset.id == dataset_id)
    res = await session.execute(stmt)
    item = res.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Caso de teste não encontrado.")
    
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(item, field, value)
        
    await session.commit()
    await session.refresh(item)
    return item


@router.delete("/datasets/{dataset_id}")
async def delete_eval_dataset_case(
    dataset_id: UUID,
    session: AsyncSession = Depends(get_db_session)
):
    """Desativa ou exclui um caso de teste do dataset de avaliações."""
    stmt = select(EvalDataset).where(EvalDataset.id == dataset_id)
    res = await session.execute(stmt)
    item = res.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Caso de teste não encontrado.")
    
    # Soft delete (ou hard delete)
    item.is_active = False
    await session.commit()
    return {"success": True, "message": "Caso de teste desativado com sucesso."}

