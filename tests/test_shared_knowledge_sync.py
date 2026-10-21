import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models import Base, PromptEngineeringDoc
from app.services.context_mcp_service import (
    read_mindflow_engineering_docs_impl,
    update_mindflow_engineering_doc_impl,
    get_generator_openai_tools,
    execute_generator_tool_call,
)


@pytest.mark.asyncio
async def test_shared_engineering_docs_persistence_and_versioning():
    """
    Valida que uma alteração feita na documentação de hipóteses (prompt_hypotheses_ledger)
    ou na memória ativa (active_memory_xml) por um agente (ex: Antigravity) é persistida
    com incremento de versão e imediatamente lida pelo outro agente (ex: PromptOps Copilot).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # 1. Leitura inicial deve retornar os documentos padrão (seed automático)
    initial = await read_mindflow_engineering_docs_impl(
        topic="prompt_hypotheses_ledger",
        _session_factory=session_factory,
    )
    assert initial["topic"] == "prompt_hypotheses_ledger"
    assert "T-01" in initial["content"]
    assert initial["version"] == 1

    # 2. Agente Antigravity adiciona nova hipótese H-09 no Livro-Razão
    new_ledger_content = initial["content"] + "\n- **Hipótese H-09 (+2 pts):** Validação cruzada Antigravity <-> PromptOps."
    updated = await update_mindflow_engineering_doc_impl(
        doc_key="prompt_hypotheses_ledger",
        content=new_ledger_content,
        updated_by="antigravity_agent",
        change_summary="Adicionada hipótese H-09 após experimento controlado",
        _session_factory=session_factory,
    )
    assert updated["doc_key"] == "prompt_hypotheses_ledger"
    assert updated["version"] == 2
    assert updated["updated_by"] == "antigravity_agent"

    # 3. Agente do PromptOps lê o documento imediatamente após e vê a H-09
    after_read = await read_mindflow_engineering_docs_impl(
        topic="prompt_hypotheses_ledger",
        _session_factory=session_factory,
    )
    assert "Hipótese H-09 (+2 pts)" in after_read["content"]
    assert after_read["version"] == 2
    assert after_read["updated_by"] == "antigravity_agent"


@pytest.mark.asyncio
async def test_generator_openai_tools_include_update_engineering_doc():
    """
    Valida que o Agente Copilot do PromptOps também possui a tool 'update_mindflow_engineering_doc'
    com instruções claras para gravar novas hipóteses ou regras na memória compartilhada.
    """
    tools = get_generator_openai_tools()
    tool_names = [t["function"]["name"] for t in tools]
    assert "read_mindflow_engineering_docs" in tool_names
    assert "update_mindflow_engineering_doc" in tool_names

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    exec_res = await execute_generator_tool_call(
        tool_name="update_mindflow_engineering_doc",
        arguments={
            "doc_key": "prompt_hypotheses_ledger",
            "content": "# Ledger Atualizado pelo PromptOps\n- H-10 testada na Arena.",
            "change_summary": "Registro de H-10 via chat do PromptOps",
        },
        context={"prompt_id": 1, "_session_factory": session_factory},
    )
    assert exec_res["output"]["version"] >= 1
    assert "H-10" in exec_res["output"]["content"]
    assert "badge" in exec_res
