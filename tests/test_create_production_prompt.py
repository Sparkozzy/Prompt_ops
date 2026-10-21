import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_create_production_prompt_validates_and_creates_supabase_row():
    mock_supabase = MagicMock()
    # Mock insert returning newly created prompt
    mock_insert_res = MagicMock()
    mock_insert_res.data = [{
        "id": 105,
        "created_at": "2026-10-21T18:00:00Z",
        "Nome do cliente": "Clínica Teste",
        "Pormpt_Name": "atendimento_triagem",
        "Prompt_Text": "<Persona>Atendente</Persona>",
        "Ligação/txt": None,
        "prompt_fup": None,
        "Prompt Insta": None
    }]
    mock_supabase.table().insert().execute.return_value = mock_insert_res

    payload = {
        "client_name": "Clínica Teste",
        "prompt_name": "atendimento_triagem",
        "channel": "whatsapp",
        "prompt_text": "<Persona>Atendente</Persona>",
        "author_email": "ryan@mindflow.ai"
    }

    with patch("app.routers.prompts.get_production_supabase", return_value=mock_supabase):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/prompts/production", json=payload)
            assert resp.status_code == 201, resp.text
            data = resp.json()
            assert data["id"] == 105
            assert data["Nome do cliente"] == "Clínica Teste"
            assert data["Pormpt_Name"] == "atendimento_triagem"
            assert data["channel"] == "whatsapp"
            assert data["Prompt_Text"] == "<Persona>Atendente</Persona>"
            assert "channels" in data
            assert "whatsapp" in data["channels"]
