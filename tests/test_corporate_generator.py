import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.schemas import MetaGeneratorRequest, MetaGeneratorResponse
from app.services.llm_service import (
    DEFAULT_CORPORATE_ACTIVE_MEMORY_XML,
    build_corporate_generator_system_prompt,
    generate_prompt_meta_suggestion,
)


def test_default_active_memory_contains_immutable_rules_and_errors():
    """Garante que a memória ativa padrão contém as regras imutáveis (R001, R002, T-01) e erros registrados (E001)."""
    assert '<memoria_ativa' in DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    assert 'escopo="prompts_de_clientes_tabela_prompts"' in DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    assert 'id="R001"' in DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    assert 'id="R002"' in DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    assert 'id="R003_T01"' in DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    assert 'id="E001"' in DEFAULT_CORPORATE_ACTIVE_MEMORY_XML


def test_build_corporate_generator_system_prompt_includes_all_phases_and_scope():
    """Garante que o meta-prompt do gerador inclui Fases 0, 1, 2, 3A, 3B e restrição de escopo a prompts de clientes."""
    sys_prompt = build_corporate_generator_system_prompt()
    assert "FASE 0" in sys_prompt
    assert "FASE 1" in sys_prompt
    assert "FASE 2" in sys_prompt
    assert "FASE 3A" in sys_prompt
    assert "FASE 3B" in sys_prompt
    assert "prompts de atendimento de clientes" in sys_prompt.lower()
    assert "R003_T01" in sys_prompt


def test_meta_generator_schemas_backward_and_forward_compatible():
    """Valida que os schemas aceitam chamadas antigas e os novos campos de memória ativa e auditoria."""
    req = MetaGeneratorRequest(
        current_prompt="<persona>Atendente</persona>",
        instruction="Adicione suporte a agendamento via MCP"
    )
    assert req.active_memory_xml is None

    res = MetaGeneratorResponse(
        suggested_prompt="<persona>Atendente v2</persona>",
        diff_unified="--- current\n+++ suggested",
        explanation="Aplicada Fase 0 a 3B",
        audit_report={"status_geral": "aprovado", "rodada_atual": 1},
        updated_active_memory_xml="<memoria_ativa versao='1.1'></memoria_ativa>"
    )
    assert res.audit_report["status_geral"] == "aprovado"
    assert "1.1" in res.updated_active_memory_xml


@pytest.mark.asyncio
async def test_generate_prompt_meta_suggestion_executes_corporate_skill():
    """Testa a execução do gerador corporativo com retorno de auditoria Fase 3B e atualização de memória."""
    mock_llm_payload = {
        "suggested_prompt": "<GlobalSettings>\nTimezone: UTC-3\n</GlobalSettings>\n<variaveis>\n[singleAccountId]\n</variaveis>",
        "explanation": "Aplicadas regras imutáveis R001, R002 e R003_T01 após absorção crítica (Fase 1) e Self-Refine (Fase 3B).",
        "audit_report": {
            "status_geral": "aprovado",
            "rodada_atual": 1,
            "imutaveis_ativas": ["R001", "R002", "R003_T01"],
            "erros_evitados": ["E001", "E002"]
        },
        "updated_active_memory_xml": DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    }

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_llm_payload)
    mock_choice.message.tool_calls = None
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("app.services.llm_service.get_openai_client", return_value=mock_client):
        result = await generate_prompt_meta_suggestion(
            current_prompt="Você é um atendente.",
            instruction="Estruture o prompt para o SAC da Kravi usando variáveis simbólicas."
        )

    assert "[singleAccountId]" in result["suggested_prompt"]
    assert result["audit_report"]["status_geral"] == "aprovado"
    assert "<memoria_ativa" in result["updated_active_memory_xml"]
    assert "--- current_prompt" in result["diff_unified"]
