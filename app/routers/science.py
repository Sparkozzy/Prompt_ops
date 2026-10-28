import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.models import (
    ScientificHypothesis,
    ExperimentPreregistration,
    ScientificExperimentRun,
    HypothesisEvent,
    PromptVersionHypothesis,
    PromptVersion,
    GlobalEvalCase,
    get_utc_now,
)
from app.schemas import (
    ScientificHypothesisCreate,
    PreregistrationCreateRequest,
    ExperimentRunStartRequest,
    ExperimentRunCompleteRequest,
    PromptVersionHypothesisCreate,
    GlobalEvalCaseCreate,
)
from app.services.science_service import (
    ensure_scientific_tables_seeded,
    compute_scientific_state_from_db,
    lint_hypothesis_generality,
    validate_and_freeze_prereg,
    evaluate_paired_replication,
)

router = APIRouter(prefix="/science", tags=["Prompt Science (M-SPEM v2)"])


@router.get("/state")
async def get_scientific_state(session: AsyncSession = Depends(get_db_session)):
    """
    Retorna o estado científico completo derivado por código (Salvaguarda S3):
    modos de falha (FM-*), hipóteses gerais (G-*) com placar calculado a partir de
    hypothesis_events, domínios/modelos validados e runs incompletos (Salvaguarda S7).
    """
    return await compute_scientific_state_from_db(session)


@router.post("/hypotheses", status_code=201)
async def create_scientific_hypothesis(
    payload: ScientificHypothesisCreate,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Registra uma nova Hipótese Geral (G-*) no Nível 2 após passar obrigatoriamente
    pelo Teste do Cliente Apagado (Salvaguarda S9 — lint_hypothesis_generality).
    """
    await ensure_scientific_tables_seeded(session)

    ok, violations = lint_hypothesis_generality(
        hypothesis_id=payload.hypothesis_id,
        title=payload.title,
        general_statement=payload.general_statement,
        scope_conditions=payload.scope_conditions,
    )
    if not ok:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Violação da Salvaguarda S9 (Teste do Cliente Apagado): enunciado contém termos específicos.",
                "violations": violations,
            },
        )

    existing = await session.execute(
        select(ScientificHypothesis).where(ScientificHypothesis.hypothesis_id == payload.hypothesis_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Hipótese '{payload.hypothesis_id}' já existe.")

    hyp = ScientificHypothesis(
        hypothesis_id=payload.hypothesis_id,
        legacy_ids=payload.legacy_ids or [],
        failure_mode_id=payload.failure_mode_id,
        title=payload.title,
        general_statement=payload.general_statement,
        scope_conditions=payload.scope_conditions,
        mechanism=payload.mechanism,
        risky_prediction=payload.risky_prediction,
        anti_cases_requirement=payload.anti_cases_requirement,
        created_by=payload.created_by,
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )
    session.add(hyp)
    await session.commit()
    await session.refresh(hyp)
    return {
        "status": "registered",
        "hypothesis_id": hyp.hypothesis_id,
        "failure_mode_id": hyp.failure_mode_id,
        "title": hyp.title,
        "initial_score": 0,
    }


@router.post("/preregister", status_code=201)
async def create_experiment_preregistration(
    payload: PreregistrationCreateRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Salvaguardas S1 e S5: Valida que a rodada testa exatamente 1 hipótese com >= 20%
    de anti-casos e critérios positivos/negativos explícitos, congelando o hash SHA-256.
    """
    await ensure_scientific_tables_seeded(session)
    prereg_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    ok, errors, frozen = validate_and_freeze_prereg(prereg_dict)
    if not ok:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Falha na validação do Pré-Registro (Salvaguardas S1/S5).",
                "errors": errors,
            },
        )

    sha = frozen["prereg_sha256"]
    existing = await session.execute(
        select(ExperimentPreregistration).where(ExperimentPreregistration.prereg_sha256 == sha)
    )
    row = existing.scalar_one_or_none()
    if not row:
        row = ExperimentPreregistration(
            prereg_sha256=sha,
            hypothesis_id=payload.hypothesis_id,
            round_mode=payload.round_mode.upper(),
            failure_mode_id=payload.failure_mode_id,
            domain_id=payload.domain_id,
            prompt_id=payload.prompt_id,
            channel=payload.channel,
            delta_p_description=payload.delta_p_description,
            model_name=payload.model_name,
            production_temperature=payload.production_temperature,
            k_samples=payload.k_samples,
            cases_json=payload.cases,
            created_by=payload.created_by,
            frozen_at=get_utc_now(),
        )
        session.add(row)
        await session.commit()

    return {
        "status": "frozen",
        "prereg_sha256": sha,
        "hypothesis_id": payload.hypothesis_id,
        "round_mode": payload.round_mode.upper(),
        "domain_id": payload.domain_id,
        "n_cases": len(payload.cases),
        "k_samples": payload.k_samples,
    }


@router.post("/runs/start", status_code=201)
async def start_scientific_experiment_run(
    payload: ExperimentRunStartRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Salvaguarda S7 (Sem Gaveta / No File-Drawer): Registra o início de um experimento
    ANTES de rodar as inferências, garantindo que runs falhos ou abortados fiquem visíveis.
    """
    await ensure_scientific_tables_seeded(session)
    run_id = payload.run_id or f"run-{uuid.uuid4().hex[:12]}"

    run_row = ScientificExperimentRun(
        run_id=run_id,
        prereg_sha256=payload.prereg_sha256,
        hypothesis_id=payload.hypothesis_id,
        round_mode=payload.round_mode.upper(),
        domain_id=payload.domain_id,
        prompt_id=payload.prompt_id,
        channel=payload.channel,
        model_name=payload.model_name,
        temperature=payload.temperature,
        status="started",
        started_at=get_utc_now(),
    )
    session.add(run_row)
    await session.commit()
    return {
        "run_id": run_id,
        "status": "started",
        "prereg_sha256": payload.prereg_sha256,
        "hypothesis_id": payload.hypothesis_id,
    }


@router.post("/runs/complete")
async def complete_scientific_experiment_run(
    payload: ExperimentRunCompleteRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Salvaguardas S2, S3 e S6: Recebe os desfechos pareados (Controle vs Tratamento),
    calcula McNemar, Beta posterior, pass^k e score_delta EXCLUSIVAMENTE por código
    (o agente não pode enviar pontuação manual) e grava o evento em hypothesis_events.
    """
    await ensure_scientific_tables_seeded(session)

    outcomes_dicts = [
        item.model_dump() if hasattr(item, "model_dump") else item.dict()
        for item in payload.paired_outcomes
    ]
    stats = evaluate_paired_replication(outcomes_dicts, k_samples=payload.k_samples)

    res_run = await session.execute(
        select(ScientificExperimentRun).where(ScientificExperimentRun.run_id == payload.run_id)
    )
    run_row = res_run.scalar_one_or_none()
    if run_row:
        run_row.status = "completed"
        run_row.control_prompt_sha256 = payload.control_prompt_sha256
        run_row.treatment_prompt_sha256 = payload.treatment_prompt_sha256
        run_row.stats_json = stats
        run_row.manifest_json = payload.raw_manifest or {}
        run_row.completed_at = get_utc_now()
    else:
        run_row = ScientificExperimentRun(
            run_id=payload.run_id,
            prereg_sha256=payload.prereg_sha256,
            hypothesis_id=payload.hypothesis_id,
            round_mode=payload.round_mode.upper(),
            domain_id=payload.domain_id,
            prompt_id=payload.prompt_id,
            channel=payload.channel,
            model_name=payload.model_name,
            temperature=payload.temperature,
            status="completed",
            control_prompt_sha256=payload.control_prompt_sha256,
            treatment_prompt_sha256=payload.treatment_prompt_sha256,
            stats_json=stats,
            manifest_json=payload.raw_manifest or {},
            started_at=get_utc_now(),
            completed_at=get_utc_now(),
        )
        session.add(run_row)

    event_id = f"evt-rep-{payload.run_id}"
    ev_row = HypothesisEvent(
        event_id=event_id,
        event_type="replication_event",
        hypothesis_id=payload.hypothesis_id,
        run_id=payload.run_id,
        prereg_sha256=payload.prereg_sha256,
        round_mode=payload.round_mode.upper(),
        domain_id=payload.domain_id,
        prompt_id=payload.prompt_id,
        channel=payload.channel,
        model_name=payload.model_name,
        temperature=payload.temperature,
        n_cases=stats["n_cases"],
        k_samples=stats["k_samples"],
        control_pass_rate=stats["control_pass_rate"],
        treatment_pass_rate=stats["treatment_pass_rate"],
        anti_case_regressions=stats["anti_case_regressions"],
        contamination_type_a=stats["contamination_type_a"],
        contamination_type_b=stats["contamination_type_b"],
        mcnemar_p=stats["mcnemar_p"],
        posterior_prob_treatment_better=stats["posterior_prob_treatment_better"],
        verdict=stats["verdict"],
        score_delta=stats["score_delta"],
        evidence_ref=f"run:{payload.run_id}",
        notes=payload.notes or stats["reason"],
        created_at=get_utc_now(),
    )
    session.add(ev_row)
    await session.commit()

    updated_state = await compute_scientific_state_from_db(session)
    hyp_state = updated_state["hypotheses"].get(payload.hypothesis_id, {})

    return {
        "run_id": payload.run_id,
        "event_id": event_id,
        "hypothesis_id": payload.hypothesis_id,
        "computed_stats": stats,
        "updated_hypothesis_score": hyp_state.get("score"),
        "updated_hypothesis_status": hyp_state.get("status"),
    }


@router.post("/provenance", status_code=201)
async def record_prompt_version_provenance(
    payload: PromptVersionHypothesisCreate,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Registra a proveniência científica de uma versão de prompt (qual hipótese G-*
    foi criada ou aplicada naquela rodada).
    """
    row = PromptVersionHypothesis(
        version_id=payload.version_id,
        prompt_id=payload.prompt_id,
        channel=payload.channel,
        hypothesis_id=payload.hypothesis_id,
        round_mode=payload.round_mode.upper(),
        delta_p_summary=payload.delta_p_summary,
        prereg_sha256=payload.prereg_sha256,
        run_id=payload.run_id,
        author_email=payload.author_email,
        created_at=get_utc_now(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return {
        "id": str(row.id),
        "version_id": str(row.version_id) if row.version_id else None,
        "prompt_id": row.prompt_id,
        "channel": row.channel,
        "hypothesis_id": row.hypothesis_id,
        "round_mode": row.round_mode,
        "delta_p_summary": row.delta_p_summary,
    }


@router.get("/impact/{hypothesis_id}")
async def get_hypothesis_impact_analysis(
    hypothesis_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Análise de Impacto: Retorna todos os prompts e versões (incluindo produção)
    que dependem da hipótese `hypothesis_id`.
    """
    stmt = (
        select(PromptVersionHypothesis)
        .where(PromptVersionHypothesis.hypothesis_id == hypothesis_id)
        .order_by(desc(PromptVersionHypothesis.created_at))
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()

    usages = []
    for r in rows:
        version_status = None
        version_label = None
        if r.version_id:
            v_res = await session.execute(
                select(PromptVersion).where(PromptVersion.id == r.version_id)
            )
            v_obj = v_res.scalar_one_or_none()
            if v_obj:
                version_status = v_obj.status
                version_label = v_obj.version_label
        usages.append({
            "prompt_id": r.prompt_id,
            "channel": r.channel,
            "version_id": str(r.version_id) if r.version_id else None,
            "version_label": version_label,
            "version_status": version_status,
            "round_mode": r.round_mode,
            "delta_p_summary": r.delta_p_summary,
            "run_id": r.run_id,
            "author_email": r.author_email,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    return {
        "hypothesis_id": hypothesis_id,
        "total_linked_versions": len(usages),
        "affected_prompts": sorted({u["prompt_id"] for u in usages}),
        "usages": usages,
    }


@router.get("/global-cases")
async def list_global_eval_cases(
    failure_mode_id: Optional[str] = Query(None),
    hypothesis_id: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db_session),
):
    stmt = select(GlobalEvalCase).order_by(desc(GlobalEvalCase.created_at))
    if failure_mode_id:
        stmt = stmt.where(GlobalEvalCase.failure_mode_id == failure_mode_id)
    if hypothesis_id:
        stmt = stmt.where(GlobalEvalCase.hypothesis_id == hypothesis_id)
    if channel:
        stmt = stmt.where(GlobalEvalCase.channel == channel)
    res = await session.execute(stmt)
    items = res.scalars().all()
    return {
        "total": len(items),
        "cases": [
            {
                "id": str(c.id),
                "failure_mode_id": c.failure_mode_id,
                "hypothesis_id": c.hypothesis_id,
                "channel": c.channel,
                "scenario_name": c.scenario_name,
                "is_anti_case": bool(c.is_anti_case),
                "is_refutation_case": bool(c.is_refutation_case),
                "input_data": c.input_data,
                "expected_criteria": c.expected_criteria,
                "expected_tool_name": c.expected_tool_name,
                "expected_tool_args": c.expected_tool_args or {},
                "forbid_tool_call": bool(c.forbid_tool_call),
                "forbidden_literal_ids": c.forbidden_literal_ids or [],
                "forbidden_symbolic_placeholders": c.forbidden_symbolic_placeholders or [],
            }
            for c in items
        ],
    }


@router.post("/global-cases", status_code=201)
async def create_global_eval_case(
    payload: GlobalEvalCaseCreate,
    session: AsyncSession = Depends(get_db_session),
):
    row = GlobalEvalCase(
        failure_mode_id=payload.failure_mode_id,
        hypothesis_id=payload.hypothesis_id,
        channel=payload.channel,
        scenario_name=payload.scenario_name,
        is_anti_case=payload.is_anti_case,
        is_refutation_case=payload.is_refutation_case,
        input_data=payload.input_data,
        expected_criteria=payload.expected_criteria,
        expected_tool_name=payload.expected_tool_name,
        expected_tool_args=payload.expected_tool_args or {},
        forbid_tool_call=payload.forbid_tool_call,
        forbidden_literal_ids=payload.forbidden_literal_ids or [],
        forbidden_symbolic_placeholders=payload.forbidden_symbolic_placeholders or [],
        created_by=payload.created_by,
        created_at=get_utc_now(),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return {
        "id": str(row.id),
        "failure_mode_id": row.failure_mode_id,
        "hypothesis_id": row.hypothesis_id,
        "scenario_name": row.scenario_name,
        "is_anti_case": row.is_anti_case,
    }
