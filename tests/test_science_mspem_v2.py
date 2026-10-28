import pytest
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models import (
    Base,
    PromptVersion,
    ExperimentPreregistration,
    ScientificExperimentRun,
    HypothesisEvent,
    PromptVersionHypothesis,
)
from app.services.eval_service import check_contamination_in_tool_calls
from app.services.science_service import (
    ensure_scientific_tables_seeded,
    lint_hypothesis_generality,
    lint_scientific_language,
    validate_and_freeze_prereg,
    evaluate_paired_replication,
    compute_scientific_state_from_db,
)


def test_s9_generality_linter_blocks_client_and_tool_names():
    """Valida Salvaguarda S9 (Teste do Cliente Apagado): bloqueia nomes de clientes, APIs e tools no enunciado geral."""
    ok_bad, violations_bad = lint_hypothesis_generality(
        hypothesis_id="G-TEST-BAD",
        title="Regra Situator Kravi",
        general_statement=(
            "Quando o cliente Kravi usa a API Situator e chama getUserByPhone, "
            "devemos passar @ no lugar de null para o ID 1277."
        ),
        scope_conditions="Apenas no WhatsApp da Kravi.",
    )
    assert ok_bad is False
    assert len(violations_bad) >= 2

    ok_good, violations_good = lint_hypothesis_generality(
        hypothesis_id="G-PRIOR-TEST",
        title="Override de Prior de Pré-Treinamento por Contraste Sintático Negativo",
        general_statement=(
            "Quando uma API ou contrato de saída exige um marcador sentinela não-padrão que contradiz "
            "o prior de pré-treinamento do modelo, combinar a proibição nominal dos valores default "
            "com a consequência técnica do erro reduz falhas de payload."
        ),
        scope_conditions="Integrações com schemas restritivos que rejeitam null/vazio.",
    )
    assert ok_good is True
    assert violations_good == []


def test_s9_scientific_language_linter_flags_triumphalist_terms():
    """Valida que adjetivos triunfalistas de 'escritor de casos de sucesso' são bloqueados."""
    ok_bad, violations_bad = lint_scientific_language(
        "A hipótese provou definitivamente e resolveu para sempre o problema com 100% infalível."
    )
    assert ok_bad is False
    assert len(violations_bad) >= 1

    ok_good, violations_good = lint_scientific_language(
        "O tratamento obteve 95.0% de aprovação pareada (McNemar p=0.0312, P(Delta>0)=0.968)."
    )
    assert ok_good is True
    assert violations_good == []


def test_s1_s5_preregistration_freezes_sha256_and_enforces_single_hypothesis_and_anti_cases():
    """Valida Salvaguardas S1 (pré-registro sha256 + 1 rodada = 1 hipótese) e S5 (>= 20% anti-casos)."""
    valid_cases_with_anti = [
        {
            "id": f"REF-{i}",
            "is_anti_case": False,
            "expected_tool": "getSchedulingAreaOptions",
            "forbidden_literal_ids": ["1277"],
        }
        for i in range(4)
    ] + [
        {
            "id": "ANTI-1",
            "is_anti_case": True,
            "forbid_tool_call": True,
        }
    ]

    # 1. Falha se tentar testar 2 hipóteses na mesma rodada
    ok_multi, errs_multi, _ = validate_and_freeze_prereg({
        "round_mode": "APPLY",
        "hypothesis_id": "G-SYMB, G-PRIOR",
        "failure_mode_id": "FM-01",
        "domain_id": "kravi_sac_whatsapp",
        "delta_p_description": "Pacote misto proibido",
        "cases": valid_cases_with_anti,
    })
    assert ok_multi is False
    assert any("1 Rodada = 1 Hipótese" in e for e in errs_multi)

    # 2. Falha se não atingir >= 20% de anti-casos
    ok_no_anti, errs_no_anti, _ = validate_and_freeze_prereg({
        "round_mode": "APPLY",
        "hypothesis_id": "G-SYMB",
        "failure_mode_id": "FM-01",
        "domain_id": "kravi_sac_whatsapp",
        "delta_p_description": "Isolar ponteiros simbólicos [var] em <variaveis>",
        "cases": [
            {
                "id": f"REF-{i}",
                "is_anti_case": False,
                "expected_tool": "getSchedulingAreaOptions",
                "forbidden_literal_ids": ["1277"],
            }
            for i in range(5)
        ],
    })
    assert ok_no_anti is False
    assert any("S5" in e for e in errs_no_anti)

    # 3. Aprova e gera sha256 determinístico quando >= 20% anti-casos e 1 única G-*
    ok_valid, errs_valid, frozen = validate_and_freeze_prereg({
        "round_mode": "APPLY",
        "hypothesis_id": "G-SYMB",
        "failure_mode_id": "FM-01",
        "domain_id": "kravi_sac_whatsapp",
        "delta_p_description": "Isolar ponteiros simbólicos [var] em <variaveis>",
        "cases": valid_cases_with_anti,
    })
    assert ok_valid is True
    assert errs_valid == []
    assert len(frozen["prereg_sha256"]) == 64


def test_s2_s3_s5_paired_evaluator_scores_passes_and_penalizes_anti_case_regression():
    """Valida que o juiz estatístico calcula McNemar, Beta-Binomial e penaliza regressão em anti-casos (-2 pts)."""
    # Caso 1: Tratamento vence Controle (5 discordâncias a favor do tratamento, 0 contra) e 0 regressão em anti-caso
    paired_win = [
        {"case_id": f"REF-{i}", "is_anti_case": False, "control_passes": 0, "treatment_passes": 3}
        for i in range(5)
    ] + [
        {"case_id": "ANTI-1", "is_anti_case": True, "control_passes": 3, "treatment_passes": 3},
        {"case_id": "ANTI-2", "is_anti_case": True, "control_passes": 3, "treatment_passes": 3},
    ]
    res_win = evaluate_paired_replication(paired_win, k_samples=3)
    assert res_win["verdict"] == "replication_pass"
    assert res_win["score_delta"] == 1
    assert res_win["posterior_prob_treatment_better"] >= 0.90

    # Caso 2: Tratamento causa falso positivo em anti-caso (Controle passava no anti-caso, Tratamento falhou)
    paired_anti_reg = [
        {"case_id": "REF-1", "is_anti_case": False, "control_passes": 0, "treatment_passes": 1},
        {"case_id": "ANTI-1", "is_anti_case": True, "control_passes": 1, "treatment_passes": 0},
    ]
    res_reg = evaluate_paired_replication(paired_anti_reg, k_samples=1)
    assert res_reg["verdict"] == "replication_fail"
    assert res_reg["score_delta"] == -2
    assert res_reg["anti_case_regressions"] == 1


def test_contamination_detector_auto_catches_bracketed_symbolic_placeholders():
    """Valida que check_contamination_in_tool_calls detecta automaticamente [var] mesmo sem lista manual."""
    is_contam, reason = check_contamination_in_tool_calls(
        tool_calls_executed=[{"tool_name": "getSchedulingAreaOptions", "input": {"accountId": "[singleAccountId]"}}],
        forbidden_literal_ids=["1277"],
        forbidden_symbolic_placeholders=[],
    )
    assert is_contam is True
    assert "Tipo B" in reason

    is_clean, reason_clean = check_contamination_in_tool_calls(
        tool_calls_executed=[{"tool_name": "getSchedulingAreaOptions", "input": {"accountId": 9941}}],
        forbidden_literal_ids=["1277"],
        forbidden_symbolic_placeholders=["[singleAccountId]", "singleAccountId"],
    )
    assert is_clean is False
    assert reason_clean is None


@pytest.mark.asyncio
async def test_science_db_seed_and_full_lifecycle():
    """
    Valida no banco SQLAlchemy:
    1. Seed inicial das 7 Failure Modes, 9 Hipóteses Gerais G-* e eventos auditados (T-01 -> G-SYMB +1 pt).
    2. Registro de pré-registro sha256, abertura de run (S7) e conclusão pareada atualizando o score.
    3. Rastreabilidade PromptVersion <-> Hipótese G-* para análise de impacto.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        await ensure_scientific_tables_seeded(session)
        state = await compute_scientific_state_from_db(session)
        assert len(state["failure_modes"]) == 7
        assert "G-SYMB" in state["hypotheses"]
        assert state["hypotheses"]["G-SYMB"]["score"] == 1
        assert state["hypotheses"]["G-LIT-ID"]["score"] == -4
        assert state["hypotheses"]["G-LIT-ID"]["status"] == "❌ X REFUTADA NO ESCOPO"

        # Simula nova replicação independente de G-SYMB em um 2º domínio (mindflow_sdr_whatsapp)
        ok_prereg, _, frozen = validate_and_freeze_prereg({
            "round_mode": "APPLY",
            "hypothesis_id": "G-SYMB",
            "failure_mode_id": "FM-01",
            "domain_id": "mindflow_sdr_whatsapp",
            "delta_p_description": "Isolar [leadId] no bloco <variaveis>",
            "cases": [
                {"id": f"C{i}", "is_anti_case": False, "expected_tool": "lookupLead", "forbidden_literal_ids": ["1277"]}
                for i in range(4)
            ] + [
                {"id": "ANTI-1", "is_anti_case": True, "forbid_tool_call": True},
            ],
        })
        assert ok_prereg is True

        prereg_row = ExperimentPreregistration(
            prereg_sha256=frozen["prereg_sha256"],
            hypothesis_id="G-SYMB",
            round_mode="APPLY",
            failure_mode_id="FM-01",
            domain_id="mindflow_sdr_whatsapp",
            prompt_id=1,
            channel="whatsapp",
            delta_p_description="Isolar [leadId] no bloco <variaveis>",
            model_name="gpt-4.1",
            production_temperature=0.5,
            k_samples=3,
            cases_json=frozen["cases"],
        )
        session.add(prereg_row)

        # S7: Inicia run com status="started" e verifica que aparece em incomplete_runs antes de fechar
        run_row = ScientificExperimentRun(
            run_id="RUN-TEST-01",
            prereg_sha256=frozen["prereg_sha256"],
            hypothesis_id="G-SYMB",
            round_mode="APPLY",
            domain_id="mindflow_sdr_whatsapp",
            prompt_id=1,
            channel="whatsapp",
            model_name="gpt-4.1",
            temperature=0.5,
            status="started",
        )
        session.add(run_row)
        await session.commit()

        state_mid = await compute_scientific_state_from_db(session)
        assert any(r["run_id"] == "RUN-TEST-01" for r in state_mid["incomplete_runs"])

        # Completa o run com vitória pareada
        run_row.status = "completed"
        evt = HypothesisEvent(
            event_id=f"evt-{uuid4().hex[:8]}",
            event_type="replication_event",
            hypothesis_id="G-SYMB",
            run_id="RUN-TEST-01",
            prereg_sha256=frozen["prereg_sha256"],
            round_mode="APPLY",
            domain_id="mindflow_sdr_whatsapp",
            prompt_id=1,
            channel="whatsapp",
            model_name="gpt-4.1",
            temperature=0.5,
            n_cases=5,
            k_samples=3,
            control_pass_rate=0.2,
            treatment_pass_rate=1.0,
            anti_case_regressions=0,
            contamination_type_a=0,
            contamination_type_b=0,
            mcnemar_p=0.0625,
            posterior_prob_treatment_better=0.9688,
            verdict="replication_pass",
            score_delta=1,
            notes="2º domínio independente validado para G-SYMB.",
        )
        session.add(evt)

        # Registra proveniência com uma PromptVersion
        pv = PromptVersion(
            prompt_id=1,
            client_id="2",
            client_name="Mindflow SDR",
            channel="whatsapp",
            version_label="v2.0.0-gsymb",
            prompt_text="<variaveis>[leadId]</variaveis>",
            model_name="gpt-4.1",
            temperature=0.5,
            status="approved",
            author_email="engenheiro@mindflow.com.br",
        )
        session.add(pv)
        await session.flush()

        pvh = PromptVersionHypothesis(
            version_id=pv.id,
            prompt_id=1,
            channel="whatsapp",
            hypothesis_id="G-SYMB",
            round_mode="APPLY",
            delta_p_summary="Isolado [leadId] no bloco <variaveis>.",
            prereg_sha256=frozen["prereg_sha256"],
            run_id="RUN-TEST-01",
        )
        session.add(pvh)
        await session.commit()

        state_final = await compute_scientific_state_from_db(session)
        assert state_final["hypotheses"]["G-SYMB"]["score"] == 2
        assert "mindflow_sdr_whatsapp" in state_final["hypotheses"]["G-SYMB"]["domains_passed"]
        assert len(state_final["incomplete_runs"]) == 0

    await engine.dispose()
