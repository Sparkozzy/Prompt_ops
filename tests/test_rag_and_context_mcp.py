import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, event
from app.models import Base, RagParentPrompt, RagChildChunk
from app.services.rag_service import (
    parse_prompt_into_parent_child_chunks,
    upsert_prompt_in_rag,
    search_rag_chunks,
    get_allowed_seed_prompt_ids,
)
from app.services.context_mcp_service import (
    prompt_ops_mcp,
    get_generator_openai_tools,
    execute_generator_tool_call,
    inspect_client_supabase_attendance_impl,
    get_arena_and_evals_context_impl,
    read_mindflow_engineering_docs_impl,
)
from app.services.llm_service import generate_prompt_meta_suggestion


def test_parse_prompt_into_parent_child_chunks_xml_and_markdown():
    """Valida a quebra modular de um prompt em Child Chunks (5 blocos ortogonais / tags XML / seções MD)."""
    sample_prompt = """<SystemPrompt>
<GlobalSettings>
Timezone: America/Sao_Paulo (UTC-3)
</GlobalSettings>
<variaveis>
[singleUserId]: ID do morador
[singleAccountId]: ID do condomínio
</variaveis>
<persona>
Você é a Sofia, assistente da Kravi.
</persona>
<ferramentas_MCP>
Use getUserByPhone silenciosamente antes de qualquer pergunta.
</ferramentas_MCP>
<few_shot_declarado>
<estado_sessao_anterior>[singleAccountId]=null</estado_sessao_anterior>
<raciocinio_interno_cod>Chamar getUserByPhone</raciocinio_interno_cod>
</few_shot_declarado>
</SystemPrompt>"""

    chunks = parse_prompt_into_parent_child_chunks(sample_prompt)
    tags = [c["section_tag"] for c in chunks]
    assert "GlobalSettings" in tags
    assert "variaveis" in tags
    assert "persona" in tags
    assert "ferramentas_MCP" in tags
    assert "few_shot_declarado" in tags
    assert len(chunks) >= 5


@pytest.mark.asyncio
async def test_rag_upsert_replaces_old_parent_and_cascades_children():
    """
    Valida que quando um novo prompt para o mesmo (prompt_id, channel) é subido no RAG,
    o Parent anterior e todos os seus Child Chunks obsoletos são excluídos automaticamente (CASCADE).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    with patch("app.services.rag_service.generate_embeddings_batch", new=AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])):
        async with session_maker() as session:
            # 1. Sobe versão v1 (antiga)
            v1_text = "<persona>Versão Antiga Obsoleta v1</persona>\n<variaveis>[oldVar]</variaveis>"
            parent_v1 = await upsert_prompt_in_rag(
                session=session,
                prompt_id=38,
                client_id="7",
                client_name="Kravi SAC",
                prompt_name="SAC v1",
                channel="whatsapp",
                prompt_text=v1_text,
                version_label="v1.0.0",
                eval_score=0.92,
            )
            v1_parent_id = parent_v1.id

            # Verifica que v1 existe com 2 chunks
            res_children_v1 = await session.execute(select(RagChildChunk))
            children_v1 = res_children_v1.scalars().all()
            assert len(children_v1) == 2
            assert any("Versão Antiga Obsoleta v1" in c.chunk_text for c in children_v1)

            # 2. Sobe versão v2 para o MESMO (prompt_id=38, channel="whatsapp")
            v2_text = "<persona>Versão Nova Atualizada v2</persona>\n<variaveis>[singleAccountId]</variaveis>"
            parent_v2 = await upsert_prompt_in_rag(
                session=session,
                prompt_id=38,
                client_id="7",
                client_name="Kravi SAC",
                prompt_name="SAC v2",
                channel="whatsapp",
                prompt_text=v2_text,
                version_label="v2.0.0",
                eval_score=0.98,
            )

            assert parent_v2.id != v1_parent_id

            # Verifica que existe APENAS 1 Parent para (38, whatsapp) e ZERO chunks da v1 obsoleta!
            res_parents = await session.execute(select(RagParentPrompt))
            all_parents = res_parents.scalars().all()
            assert len(all_parents) == 1
            assert all_parents[0].version_label == "v2.0.0"

            res_children_v2 = await session.execute(select(RagChildChunk))
            children_v2 = res_children_v2.scalars().all()
            assert len(children_v2) == 2
            assert all(c.parent_id == parent_v2.id for c in children_v2)
            assert not any("Versão Antiga Obsoleta v1" in c.chunk_text for c in children_v2)
            assert any("Versão Nova Atualizada v2" in c.chunk_text for c in children_v2)

    await engine.dispose()


@pytest.mark.asyncio
async def test_rag_semantic_and_filtered_search():
    """Valida a busca semântica e por filtro de section_tag/channel no RAG Parent-Child."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Embeddings ortogonais simulados: chunk 0 = [1.0, 0.0], chunk 1 = [0.0, 1.0]
    with patch("app.services.rag_service.generate_embeddings_batch", new=AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])):
        async with session_maker() as session:
            text = (
                "<ferramentas_MCP>Use check_availability e schedule_appointment para agendar reunião.</ferramentas_MCP>\n"
                "<few_shot_declarado>Exemplo de contorno de objeção de preço no treinamento SDR.</few_shot_declarado>"
            )
            await upsert_prompt_in_rag(
                session=session,
                prompt_id=32,
                client_id="sdr",
                client_name="Treinamento SDR 5.1",
                prompt_name="SDR",
                channel="ligacao",
                prompt_text=text,
                version_label="prod-ligacao-1",
                eval_score=0.95,
            )

    # Query buscando o segundo chunk ([0.0, 1.0])
    with patch("app.services.rag_service.generate_ embeddings_batch" if False else "app.services.rag_service.generate_embeddings_batch", new=AsyncMock(return_value=[[0.0, 1.0]])):
        async with session_maker() as session:
            results = await search_rag_chunks(
                session=session,
                query="objeção de preço SDR",
                section_tag="few_shot_declarado",
                channel="ligacao",
                top_k=2,
            )
            assert len(results) == 1
            assert results[0]["section_tag"] == "few_shot_declarado"
            assert results[0]["parent"]["client_name"] == "Treinamento SDR 5.1"
            assert "objeção de preço" in results[0]["chunk_text"]

    await engine.dispose()


def test_allowed_seed_prompt_ids_only_active_clients_and_sdr():
    """Garante que o filtro de seed inclui apenas clientes ativos (client_configurations) + Treinamento SDR (32, 67)."""
    mock_configs = [
        {"client_id": "2", "client_name": "Mindflow", "prompt_id": 1},
        {"client_id": "7", "client_name": "Kravi SAC", "prompt_id": 38},
        {"client_id": "9", "client_name": "Versa", "prompt_id": None},
    ]
    mock_prompts = [
        {"id": 1, "Nome do cliente": "Mindflow"},
        {"id": 15, "Nome do cliente": "mindflow form 2 - obsoleto"},
        {"id": 32, "Nome do cliente": "Treinamento SDR 5.1"},
        {"id": 38, "Nome do cliente": "Kravi SAC"},
        {"id": 41, "Nome do cliente": "Versa"},
        {"id": 67, "Nome do cliente": " Treinamento SDR"},
        {"id": 100, "Nome do cliente": "test"},
    ]
    allowed = get_allowed_seed_prompt_ids(mock_configs, mock_prompts)
    assert 1 in allowed
    assert 38 in allowed
    assert 41 in allowed  # Versa (active client_name match)
    assert 32 in allowed  # Treinamento SDR 5.1
    assert 67 in allowed  # Treinamento SDR
    assert 15 not in allowed  # Obsoleto
    assert 100 not in allowed  # Teste solto


def test_context_mcp_tools_have_detailed_instructions():
    """Valida que as 5 tools do servidor MCP possuem descrições detalhadas orientando quando e como usá-las."""
    tools = get_generator_openai_tools()
    tool_names = {t["function"]["name"] for t in tools}
    expected = {
        "search_rag_prompt_blocks",
        "inspect_client_supabase_attendance",
        "get_arena_and_evals_context",
        "inspect_client_mcp_schemas",
        "read_mindflow_engineering_docs",
    }
    assert expected.issubset(tool_names)
    for t in tools:
        desc = t["function"]["description"]
        assert len(desc) >= 120, f"Tool {t['function']['name']} precisa de instruções detalhadas no MCP."


@pytest.mark.asyncio
async def test_inspect_client_supabase_attendance_multi_filter():
    """Valida que a tool de inspeção de atendimento consulta memória (n8n_chat_histories / Retell) e steps de tools no Supabase do cliente."""
    res = await inspect_client_supabase_attendance_impl(
        prompt_id=38,
        phone_or_session="+5548996027108",
        execution_id=None,
        query_mode="by_phone",
        _mock_data={
            "client_name": "Kravi SAC",
            "chat_memory": [
                {"session_id": "5548996027108", "role": "human", "content": "Quero reservar o salão"},
                {"session_id": "5548996027108", "role": "ai", "content": "Qual unidade?"},
            ],
            "workflow_steps": [
                {
                    "step_name": "whatsapp_flow_mcp_connection",
                    "status": "SUCCESS",
                    "input_data": {"tool": "getUserByPhone", "phone": "5548996027108"},
                    "output_data": {"singleAccountId": 1277, "singleUserId": 275983},
                }
            ],
        },
    )
    assert res["client_name"] == "Kravi SAC"
    assert len(res["chat_memory"]) == 2
    assert res["workflow_steps"][0]["step_name"] == "whatsapp_flow_mcp_connection"


@pytest.mark.asyncio
async def test_generate_prompt_meta_suggestion_on_demand_tool_calling():
    """
    Valida que o Agente Gerador decide chamar tools on-demand (ex: get_arena_and_evals_context e search_rag_prompt_blocks)
    quando solicitado na instrução, e retorna a lista tools_consulted com badges para o frontend.
    """
    # 1ª resposta do LLM: decide acionar get_arena_and_evals_context
    tc_mock = MagicMock()
    tc_mock.id = "call_arena_123"
    tc_mock.function.name = "get_arena_and_evals_context"
    tc_mock.function.arguments = json.dumps({"include_arena_chat": True, "include_eval_failures": True})

    first_choice = MagicMock()
    first_choice.message.content = None
    first_choice.message.tool_calls = [tc_mock]
    first_resp = MagicMock()
    first_resp.choices = [first_choice]

    # 2ª resposta do LLM: após receber o resultado da tool, gera o JSON final das Fases 0 a 3B
    final_payload = {
        "suggested_prompt": "<SystemPrompt>\n<persona>Corrigido com base na conversa da Arena A/B</persona>\n</SystemPrompt>",
        "explanation": "Analisei a conversa atual na Arena A/B via tool on-demand e ajustei a persona.",
        "audit_report": {
            "status_geral": "aprovado",
            "rodada_atual": 1,
            "imutaveis_ativas": ["R001", "R002", "R003_T01"],
            "erros_evitados": ["E001", "E002"],
        },
        "updated_active_memory_xml": "<memoria_ativa versao=\"1.0\"></memoria_ativa>",
    }
    second_choice = MagicMock()
    second_choice.message.content = json.dumps(final_payload)
    second_choice.message.tool_calls = None
    second_resp = MagicMock()
    second_resp.choices = [second_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[first_resp, second_resp])

    with patch("app.services.llm_service.get_openai_client", return_value=mock_client):
        result = await generate_prompt_meta_suggestion(
            current_prompt="<SystemPrompt><persona>Antigo</persona></SystemPrompt>",
            instruction="Leia minha conversa atual no teste A/B e aplique mudanças.",
            arena_history={
                "A": [{"role": "user", "content": "Oi"}, {"role": "assistant", "content": "Resposta longa demais"}],
                "B": [{"role": "user", "content": "Oi"}, {"role": "assistant", "content": "Oi! Como posso ajudar?"}],
            },
            prompt_id=38,
            client_id="7",
            channel="whatsapp",
        )

    assert "Corrigido com base na conversa da Arena A/B" in result["suggested_prompt"]
    assert len(result["tools_consulted"]) == 1
    assert result["tools_consulted"][0]["tool_name"] == "get_arena_and_evals_context"
    assert "Arena" in result["tools_consulted"][0]["label"]
