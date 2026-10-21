from unittest.mock import AsyncMock, patch
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models import Base, PromptVersion
from app.services.context_mcp_service import (
    list_prompt_eval_datasets_impl,
    manage_prompt_eval_case_impl,
    run_prompt_eval_suite_impl,
    get_generator_openai_tools,
    execute_generator_tool_call,
)
from app.services.llm_service import build_corporate_generator_system_prompt


@pytest.mark.asyncio
async def test_copilot_evals_tools_crud_and_scientific_fields():
    """
    Valida que o Agente do PromptOps (e do Antigravity) pode criar, listar, editar e excluir
    casos de Evals seguindo todas as regras científicas do prompt_engineer.md:
    - scenario_name, user_message, expected_criteria
    - requires_tool, expected_tool_name
    - is_refutation_case, forbidden_literal_ids (Tipo A), forbidden_symbolic_placeholders (Tipo B)
    - auditoria (created_by, updated_by, deleted_by)
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # 1. Cria um Caso de Refutação Científica (Popperiano)
    created = await manage_prompt_eval_case_impl(
        action="create",
        prompt_id=38,
        client_id="38",
        channel="whatsapp",
        scenario_name="REF-01: Novo morador Carlos no Condomínio Solar (Anti-Contaminação Tipo A/B)",
        user_message="Oi, sou o Carlos e quero reservar a churrasqueira amanhã.",
        expected_criteria="Deve acionar getUserByPhone ou getSchedulingAreaOptions sem usar IDs antigos (1277, 275983) e sem enviar '[singleAccountId]' literal.",
        requires_tool=True,
        expected_tool_name="getUserByPhone",
        is_refutation_case=True,
        forbidden_literal_ids=["1277", "275983"],
        forbidden_symbolic_placeholders=["[singleAccountId]", "singleAccountId"],
        author_email="prompt_ops_copilot",
        _session_factory=session_factory,
    )
    assert created["action"] == "created"
    case_id = created["case"]["id"]
    assert created["case"]["is_refutation_case"] is True
    assert "1277" in created["case"]["forbidden_literal_ids"]
    assert "[singleAccountId]" in created["case"]["forbidden_symbolic_placeholders"]
    assert created["case"]["created_by"] == "prompt_ops_copilot"

    # 2. Lista os casos ativos do prompt 38
    listed = await list_prompt_eval_datasets_impl(
        prompt_id=38,
        channel="whatsapp",
        _session_factory=session_factory,
    )
    assert listed["total_cases"] == 1
    assert listed["cases"][0]["id"] == case_id

    # 3. Atualiza a rubrica do caso
    updated = await manage_prompt_eval_case_impl(
        action="update",
        case_id=case_id,
        expected_criteria="Rubrica atualizada com validação estrita de CoD.",
        author_email="prompt_ops_copilot",
        _session_factory=session_factory,
    )
    assert updated["action"] == "updated"
    assert "CoD" in updated["case"]["expected_criteria"]

    # 4. Soft-delete com auditoria
    deleted = await manage_prompt_eval_case_impl(
        action="delete",
        case_id=case_id,
        author_email="prompt_ops_copilot",
        _session_factory=session_factory,
    )
    assert deleted["action"] == "deleted"

    listed_after = await list_prompt_eval_datasets_impl(
        prompt_id=38,
        channel="whatsapp",
        _session_factory=session_factory,
    )
    assert listed_after["total_cases"] == 0


@pytest.mark.asyncio
async def test_copilot_run_prompt_eval_suite_updates_score_and_returns_report():
    """
    Valida que o Agente do PromptOps consegue disparar a bateria de Evals (run_prompt_eval_suite),
    avaliando os casos cadastrados (incluindo detecção de contaminação e LLM Judge) e atualizando
    o eval_score da versão selecionada.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Cria uma versão de prompt para testar
    async with session_factory() as session:
        v = PromptVersion(
            prompt_id=38,
            client_id="38",
            client_name="Kravi",
            channel="whatsapp",
            version_label="v1.1.0-draft",
            prompt_text="<identidade>Assistente Kravi</identidade>",
            model_name="gpt-4.1",
            temperature=0.7,
            status="draft",
            author_email="engenheiro@mindflow.com.br",
        )
        session.add(v)
        await session.commit()
        await session.refresh(v)
        version_id = str(v.id)

    # Cadastra 1 caso de teste
    await manage_prompt_eval_case_impl(
        action="create",
        prompt_id=38,
        client_id="38",
        channel="whatsapp",
        scenario_name="Saudação inicial e identificação",
        user_message="Olá, bom dia!",
        expected_criteria="Deve cumprimentar educadamente como Assistente Kravi.",
        _session_factory=session_factory,
    )

    mock_eval_res = {
        "passed": True,
        "is_contaminated": False,
        "contamination_reason": None,
        "generated_response": "Olá, bom dia! Sou a assistente virtual da Kravi, como posso ajudar?",
        "judge_reasoning": "Cumprimentou educadamente conforme a rubrica.",
        "tool_called": None,
        "tool_call_valid": None,
        "execution_time_ms": 420,
    }

    with patch("app.services.context_mcp_service.evaluate_single_turn_case", new=AsyncMock(return_value=mock_eval_res)):
        run_report = await execute_generator_tool_call(
            tool_name="run_prompt_eval_suite",
            arguments={"test_live_editor_prompt": True},
            context={
                "prompt_id": 38,
                "client_id": "38",
                "channel": "whatsapp",
                "version_id": version_id,
                "current_prompt": "<identidade>Assistente Kravi Atualizada</identidade>",
                "_session_factory": session_factory,
            },
        )

    out = run_report["output"]
    assert out["total_cases"] == 1
    assert out["passed_cases"] == 1
    assert out["approval_rate"] == 1.0
    assert out["quality_gate_passed"] is True


def test_generator_system_prompt_includes_scientific_eval_rules():
    """
    Valida que o System Prompt do Agente do PromptOps incorpora as regras científicas
    do prompt_engineer.md para criação de Casos de Refutação e execução de Evals.
    """
    sys_prompt = build_corporate_generator_system_prompt()
    assert "list_prompt_eval_datasets" in sys_prompt
    assert "manage_prompt_eval_case" in sys_prompt
    assert "run_prompt_eval_suite" in sys_prompt
    assert "is_refutation_case" in sys_prompt
    assert "90%" in sys_prompt

    tools = get_generator_openai_tools()
    tool_names = [t["function"]["name"] for t in tools]
    assert "list_prompt_eval_datasets" in tool_names
    assert "manage_prompt_eval_case" in tool_names
    assert "run_prompt_eval_suite" in tool_names
