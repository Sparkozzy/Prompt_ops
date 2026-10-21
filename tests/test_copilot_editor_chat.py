import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.schemas import MetaGeneratorRequest, MetaGeneratorResponse
from app.services.llm_service import generate_prompt_meta_suggestion


def test_copilot_schemas_support_chat_history_and_prompt_modified():
    """Valida que MetaGeneratorRequest aceita chat_history e MetaGeneratorResponse expõe prompt_modified."""
    req = MetaGeneratorRequest(
        current_prompt="<persona>Assistente</persona>",
        instruction="Quais tools o cliente usa?",
        chat_history=[
            {"role": "user", "content": "Oi"},
            {"role": "assistant", "content": "Olá! Como posso ajudar no prompt?"},
        ],
    )
    assert len(req.chat_history) == 2

    res = MetaGeneratorResponse(
        suggested_prompt="<persona>Assistente</persona>",
        prompt_modified=False,
        diff_unified="Nenhuma diferença textual detectada.",
        explanation="O cliente usa check_availability e schedule_appointment.",
    )
    assert res.prompt_modified is False


@pytest.mark.asyncio
async def test_copilot_inquiry_only_does_not_modify_prompt():
    """
    Valida que quando o usuário faz apenas uma pergunta/consulta no chat (ex: consultar MCP/RAG),
    o agente responde no chat sem alterar o texto do prompt (prompt_modified=False).
    """
    original_prompt = "<SystemPrompt>\n<persona>Sofia da Kravi</persona>\n</SystemPrompt>"
    mock_llm_payload = {
        "prompt_modified": False,
        "suggested_prompt": original_prompt,
        "explanation": "Consultei os schemas MCP do cliente: ele possui getUserByPhone e createScheduling.",
        "audit_report": {"status_geral": "consulta_informativa", "rodada_atual": 1},
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
            current_prompt=original_prompt,
            instruction="Quais ferramentas MCP esse cliente tem disponíveis?",
            chat_history=[{"role": "user", "content": "Olá"}],
        )

    assert result["prompt_modified"] is False
    assert result["suggested_prompt"] == original_prompt
    assert "getUserByPhone" in result["explanation"]


@pytest.mark.asyncio
async def test_copilot_realtime_edit_sets_prompt_modified_true():
    """
    Valida que quando o usuário pede uma alteração no prompt durante o chat,
    o agente retorna prompt_modified=True, o novo suggested_prompt e o diff_unified para atualizar o editor em tempo real.
    """
    original_prompt = "<SystemPrompt>\n<persona>Sofia da Kravi</persona>\n</SystemPrompt>"
    updated_prompt = "<SystemPrompt>\n<persona>Sofia, especialista de atendimento da Kravi (máx 200 caracteres)</persona>\n</SystemPrompt>"
    mock_llm_payload = {
        "prompt_modified": True,
        "suggested_prompt": updated_prompt,
        "explanation": "Atualizei a tag <persona> adicionando o limite de 200 caracteres.",
        "audit_report": {"status_geral": "aprovado", "rodada_atual": 1},
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
            current_prompt=original_prompt,
            instruction="Adicione limite de 200 caracteres na persona.",
            chat_history=[
                {"role": "user", "content": "Quais tools temos?"},
                {"role": "assistant", "content": "Temos getUserByPhone."},
            ],
        )

    assert result["prompt_modified"] is True
    assert result["suggested_prompt"] == updated_prompt
    assert "+<persona>Sofia, especialista" in result["diff_unified"]
