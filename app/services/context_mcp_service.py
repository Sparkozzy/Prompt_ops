import json
import re
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from supabase import create_client
from app.database import async_session_maker, get_production_supabase
from app.models import EvalRun, EvalCaseResult, ToolExecutionLog
from app.services.rag_service import search_rag_chunks
from app.services.mcp_service import get_client_mcp_urls, get_client_tools_and_map

# Instância oficial FastMCP expondo instruções ricas e detalhadas para Agentes
prompt_ops_mcp = FastMCP(
    "prompt_ops_context_mcp",
    instructions=(
        "Servidor MCP oficial de Contexto, RAG Parent-Child e Diagnóstico Multi-Tenant da MindFlow (PromptOps). "
        "Utilize estas ferramentas SOB DEMANDA (on-demand) sempre que precisar consultar: "
        "(1) Exemplos e blocos modulares de outros prompts de clientes ativos ou de Treinamento SDR no RAG Parent-Child; "
        "(2) Memória conversacional (n8n_chat_histories / Retell) e rastreio de execução de tools (workflow_step_executions) "
        "de um atendimento específico no Supabase isolado do cliente; "
        "(3) A conversa atual que está ocorrendo na Arena A/B ou falhas recentes nos Evals; "
        "(4) Os schemas exatos das ferramentas MCP habilitadas para o cliente; "
        "(5) Documentação oficial de Engenharia de Prompts (Livro-Razão de Hipóteses T-01/H-04/H-03/H-01 e Guia Supabase)."
    ),
)


ENGINEERING_DOCS_LIBRARY: Dict[str, str] = {
    "prompt_hypotheses_ledger": """# Livro-Razão Científico de Hipóteses (Prompt Science — MindFlow)
- **Teoria Consolidada T-01 (8 pts — OBRIGATÓRIA):** Isolamento de Estado por Variáveis Simbólicas `[var]` + Few-Shot Estruturado com Chain of Draft (`<raciocinio_interno_cod>`). Todo prompt que aciona ferramentas MCP deve declarar `<variaveis>` com ponteiros simbólicos (`[singleUserId]`, `[singleAccountId]`, `[unitId]`, `[selectedArea]`, etc.) e NUNCA usar IDs literais de casos antigos nos exemplos.
- **Hipótese H-04 (+3 pts — Recomendada para Situator/Kravi):** Protocolo de Preenchimento por Contraste Sintático (`@` vs `null`). Em APIs que rejeitam `null` ou string vazia, instruir explicitamente o uso de `"@"` para campos não aplicáveis.
- **Hipótese H-03 (+1 pt — Guardrail Anti-Contaminação Tipo B):** Re-hidratação Silenciosa de Estado. Se o histórico não contiver o ID numérico resolvido, é proibido enviar o nome literal do placeholder (`"singleAccountId"`) para a tool; o agente deve disparar silenciosamente a tool de identificação (`getUserByPhone`) primeiro.
- **Hipótese BANIDA H-01 (-4 pts — ❌ X REFUTADA):** Few-Shot com IDs Literais de Casos Antigos (ex: `1277`, `275983`, `15799`). Causa Contaminação Tipo A (vazamento cruzado entre clientes). Terminantemente proibido.""",
    "canonical_prompt_anatomy": """# Anatomia Modular Canônica do Prompt (5 Blocos Ortogonais)
1. `<GlobalSettings>`: Timezone (`America/Sao_Paulo` UTC-3), idioma, modo de raciocínio interno (`Chain of Draft - CoD`).
2. `<variaveis>`: Declaração tipada de ponteiros de sessão simbólicos (`[singleUserId]`, `[singleAccountId]`, `[customer_name]`). Zero números literais fixos.
3. `<persona>` e `<regras_comunicacao>`: Identidade, autoridade, limite de caracteres (ex: máx 200 caracteres no WhatsApp) e compliance.
4. `<ferramentas_MCP>` e `<fluxo_de_uso>`: Máquina de estados determinística (qual tool chamar silenciosamente, pré-requisitos e tratamento de retorno).
5. `<few_shot_declarado>`: Exemplos contendo `<estado_sessao_anterior>`, `<raciocinio_interno_cod>`, `<chamada_ferramenta>` e `<estado_sessao_atualizado>` usando exclusivamente variáveis simbólicas `[var]`.""",
    "supabase_data_guide": """# Guia de Estrutura de Dados Multi-Tenant (Supabase Master & Cliente)
- **Supabase Master (`client_configurations`):** Mapeia `client_id`, `prompt_id`, `llm_model`, `llm_temperature`, `mcp_urls`, `supabase_url` e `supabase_service_key` do tenant isolado.
- **Tabela `Prompts`:** Colunas por canal: `Prompt_Text` (WhatsApp), `"Ligação/txt"` (Voz/Retell), `prompt_fup` (Follow-Up), `"Prompt Insta"` (Instagram).
- **Supabase do Cliente (Isolado):**
  - `n8n_chat_histories`: Memória multi-turn do WhatsApp (`session_id` = telefone, `message` JSON com `type: human/ai` e `content`).
  - `Retell_calls_Mindflow`: Transcrições de chamadas de voz (`to_number`, `transcript`, `call_summary`, `disconnection_reason`, `Marcada`).
  - `workflow_executions` & `workflow_step_executions`: Rastreabilidade EDW de cada passo e chamada de tool MCP (`step_name`, `status`, `input_data`, `output_data`, `error_details`).""",
}


async def search_rag_prompt_blocks_impl(
    query: str,
    section_tag: Optional[str] = None,
    channel: Optional[str] = None,
    exclude_prompt_id: Optional[int] = None,
    include_parent_text: bool = False,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Implementação da busca no RAG Parent-Child de prompts de clientes ativos e Treinamento SDR."""
    async with async_session_maker() as session:
        results = await search_rag_chunks(
            session=session,
            query=query,
            section_tag=section_tag,
            channel=channel,
            exclude_prompt_id=exclude_prompt_id,
            include_parent_text=include_parent_text,
            top_k=top_k,
        )
    return {
        "query": query,
        "section_tag_filter": section_tag,
        "channel_filter": channel,
        "total_results": len(results),
        "chunks": results,
    }


def _sanitize_digits(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


async def inspect_client_supabase_attendance_impl(
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    phone_or_session: Optional[str] = None,
    execution_id: Optional[str] = None,
    query_mode: str = "auto",
    limit: int = 20,
    _mock_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Implementação da consulta on-demand ao ambiente Supabase isolado do cliente
    para inspecionar memória conversacional (n8n_chat_histories / Retell_calls_Mindflow)
    e uso de tools/steps (workflow_executions / workflow_step_executions).
    """
    if _mock_data is not None:
        return _mock_data

    master_supa = get_production_supabase()
    client_cfg = None

    if prompt_id is not None:
        res = master_supa.table("client_configurations").select("*").eq("prompt_id", int(prompt_id)).execute()
        if res.data:
            client_cfg = res.data[0]
    if not client_cfg and client_id is not None:
        res = master_supa.table("client_configurations").select("*").eq("client_id", str(client_id)).execute()
        if res.data:
            client_cfg = res.data[0]

    # Se o cliente tiver um Supabase próprio configurado em client_configurations, conecta nele; senão usa o Master
    tenant_supa = master_supa
    client_name = "Master / Geral"
    if client_cfg:
        client_name = client_cfg.get("client_name") or f"Client {client_cfg.get('client_id')}"
        c_url = client_cfg.get("supabase_url")
        c_key = client_cfg.get("supabase_service_key") or client_cfg.get("supabase_anon_key")
        if c_url and c_key:
            try:
                tenant_supa = create_client(c_url, c_key)
            except Exception as e:
                print(f"[Context MCP Warning] Falha ao conectar ao Supabase do tenant {client_name}: {e}")

    chat_memory: List[Dict[str, Any]] = []
    voice_calls: List[Dict[str, Any]] = []
    workflow_traces: List[Dict[str, Any]] = []
    workflow_steps: List[Dict[str, Any]] = []

    digits = _sanitize_digits(phone_or_session) if phone_or_session else ""

    # 1. Consulta memória de chat (n8n_chat_histories)
    try:
        q_hist = tenant_supa.table("n8n_chat_histories").select("*").order("id", desc=True).limit(limit)
        if digits:
            # Busca pelo final do número (últimos 8 ou 9 dígitos) para cobrir variações com/sem +55 ou 9º dígito
            tail = digits[-8:] if len(digits) >= 8 else digits
            q_hist = tenant_supa.table("n8n_chat_histories").select("*").ilike("session_id", f"%{tail}%").order("id", desc=True).limit(limit)
        hist_res = q_hist.execute()
        for row in reversed(hist_res.data or []):
            msg_obj = row.get("message") or {}
            chat_memory.append({
                "id": row.get("id"),
                "session_id": row.get("session_id"),
                "role": msg_obj.get("type", "unknown"),
                "content": str(msg_obj.get("content", ""))[:1200],
                "tool_calls": msg_obj.get("tool_calls") or msg_obj.get("additional_kwargs", {}).get("tool_calls"),
            })
    except Exception as e:
        print(f"[Context MCP] Info n8n_chat_histories: {e}")

    # 2. Consulta chamadas de voz (Retell_calls_Mindflow) se aplicável
    try:
        q_retell = tenant_supa.table("Retell_calls_Mindflow").select(
            "id, created_at, to_number, disconnection_reason, call_summary, transcript, Marcada"
        ).order("created_at", desc=True).limit(5)
        if digits:
            tail = digits[-8:] if len(digits) >= 8 else digits
            q_retell = tenant_supa.table("Retell_calls_Mindflow").select(
                "id, created_at, to_number, disconnection_reason, call_summary, transcript, Marcada"
            ).ilike("to_number", f"%{tail}%").order("created_at", desc=True).limit(5)
        retell_res = q_retell.execute()
        for r in (retell_res.data or []):
            voice_calls.append({
                "id": r.get("id"),
                "created_at": r.get("created_at"),
                "to_number": r.get("to_number"),
                "disconnection_reason": r.get("disconnection_reason"),
                "call_summary": r.get("call_summary"),
                "transcript": str(r.get("transcript") or "")[:2000],
                "scheduled": r.get("Marcada"),
            })
    except Exception as e:
        print(f"[Context MCP] Info Retell_calls_Mindflow: {e}")

    # 3. Consulta Rastreabilidade EDW (workflow_executions & workflow_step_executions)
    try:
        target_exec_ids: List[str] = []
        if execution_id:
            target_exec_ids.append(execution_id.strip())
        else:
            q_exec = tenant_supa.table("workflow_executions").select(
                "id, workflow_name, status, input_data, output_data, error_details, created_at"
            ).order("created_at", desc=True).limit(15)
            if query_mode == "latest_errors":
                q_exec = tenant_supa.table("workflow_executions").select(
                    "id, workflow_name, status, input_data, output_data, error_details, created_at"
                ).eq("status", "FAILED").order("created_at", desc=True).limit(10)
            exec_res = q_exec.execute()
            for ex in (exec_res.data or []):
                inp_str = json.dumps(ex.get("input_data") or {}, ensure_ascii=False)
                if digits and digits[-8:] not in inp_str:
                    continue
                workflow_traces.append(ex)
                target_exec_ids.append(ex["id"])
                if len(target_exec_ids) >= 5:
                    break

        if target_exec_ids:
            steps_res = tenant_supa.table("workflow_step_executions").select(
                "execution_id, step_name, status, input_data, output_data, error_details, started_at"
            ).in_("execution_id", target_exec_ids).order("started_at", desc=False).limit(30).execute()
            workflow_steps = steps_res.data or []
    except Exception as e:
        print(f"[Context MCP] Info workflow_step_executions: {e}")

    return {
        "client_name": client_name,
        "prompt_id": prompt_id,
        "filter_applied": {
            "phone_or_session": phone_or_session,
            "execution_id": execution_id,
            "query_mode": query_mode,
        },
        "chat_memory": chat_memory,
        "voice_calls": voice_calls,
        "workflow_traces": workflow_traces,
        "workflow_steps": workflow_steps,
    }


async def get_arena_and_evals_context_impl(
    arena_history: Optional[Dict[str, Any]] = None,
    failing_examples: Optional[List[Dict[str, Any]]] = None,
    version_id: Optional[Any] = None,
    prompt_id: Optional[int] = None,
    include_arena_chat: bool = True,
    include_eval_failures: bool = True,
) -> Dict[str, Any]:
    """
    Retorna sob demanda a conversa atual da Arena A/B (Versão A vs Versão B),
    os logs de tools executadas recentemente na Arena e as falhas recentes da bateria de Evals.
    """
    result: Dict[str, Any] = {}

    if include_arena_chat:
        result["arena_conversation_ab"] = arena_history or {"A": [], "B": []}

    if failing_examples:
        result["provided_failing_examples"] = failing_examples

    if include_eval_failures:
        eval_failures = []
        recent_tool_logs = []
        try:
            async with async_session_maker() as session:
                # Busca última bateria de evals da versão ou do prompt
                stmt_run = select(EvalRun).options(selectinload(EvalRun.cases)).order_by(desc(EvalRun.started_at)).limit(1)
                if version_id:
                    stmt_run = stmt_run.where(EvalRun.version_id == version_id)
                elif prompt_id:
                    stmt_run = stmt_run.where(EvalRun.prompt_id == int(prompt_id))

                res_run = await session.execute(stmt_run)
                last_run = res_run.scalar_one_or_none()
                if last_run:
                    for c in last_run.cases:
                        if not c.passed:
                            eval_failures.append({
                                "dataset_id": str(c.dataset_id),
                                "generated_response": c.generated_response,
                                "judge_reasoning": c.judge_reasoning,
                                "tool_called": c.tool_called,
                                "tool_call_valid": c.tool_call_valid,
                            })

                # Busca últimos logs de tools executadas no PromptOps
                stmt_tools = select(ToolExecutionLog).order_by(desc(ToolExecutionLog.created_at)).limit(10)
                if version_id:
                    stmt_tools = stmt_tools.where(ToolExecutionLog.version_id == version_id)
                res_tools = await session.execute(stmt_tools)
                for tl in res_tools.scalars().all():
                    recent_tool_logs.append({
                        "tool_name": tl.tool_name,
                        "input_payload": tl.input_payload,
                        "output_payload": tl.output_payload,
                        "status": tl.status,
                        "latency_ms": tl.latency_ms,
                    })
        except Exception as e:
            print(f"[Context MCP Warning] Falha ao ler eval_failures/tool_logs locais: {e}")

        result["recent_eval_failures"] = eval_failures
        result["recent_arena_tool_execution_logs"] = recent_tool_logs

    return result


async def inspect_client_mcp_schemas_impl(
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Descobre e inspeciona os JSON Schemas exatos das ferramentas MCP reais do cliente."""
    mcp_urls = await get_client_mcp_urls(prompt_id=prompt_id, client_id=client_id)
    tools, tool_map = await get_client_tools_and_map(mcp_urls)
    return {
        "mcp_urls": mcp_urls,
        "total_tools": len(tools),
        "tools_schemas": tools,
        "tool_to_server_map": tool_map,
    }


def read_mindflow_engineering_docs_impl(topic: str = "all") -> Dict[str, Any]:
    """Retorna a documentação oficial de Engenharia de Prompts e Estrutura de Dados da MindFlow."""
    t = (topic or "all").strip().lower()
    if t in ENGINEERING_DOCS_LIBRARY:
        return {"topic": t, "content": ENGINEERING_DOCS_LIBRARY[t]}
    return {
        "topic": "all",
        "documents": ENGINEERING_DOCS_LIBRARY,
    }


# ============================================================================
# Registro das Ferramentas no FastMCP Server (para clientes SSE externos)
# ============================================================================

@prompt_ops_mcp.tool(
    name="search_rag_prompt_blocks",
    description=(
        "QUANDO USAR: Sempre que o usuário pedir para consultar/usar outro prompt como referência/exemplo, "
        "ou quando você precisar ver como os melhores prompts de clientes ativos e de Treinamento SDR da MindFlow "
        "estruturam blocos específicos (<GlobalSettings>, <variaveis>, <persona>, <ferramentas_MCP>, <few_shot_declarado>, "
        "contorno de objeções de SDR ou agendamento). "
        "COMO USAR: Passe 'query' descrevendo o padrão desejado (ex: 'agendamento check_availability few shot' ou 'objeção preço SDR'), "
        "opcionalmente filtre por 'section_tag' (ex: 'few_shot_declarado', 'ferramentas_MCP', 'variaveis', 'Persona') e 'channel' ('whatsapp', 'ligacao', 'fup'). "
        "Use 'include_parent_text=True' apenas se precisar ler o prompt pai inteiro além do bloco filho."
    ),
)
async def mcp_search_rag_prompt_blocks(
    query: str,
    section_tag: Optional[str] = None,
    channel: Optional[str] = None,
    exclude_prompt_id: Optional[int] = None,
    include_parent_text: bool = False,
    top_k: int = 4,
) -> str:
    res = await search_rag_prompt_blocks_impl(
        query=query,
        section_tag=section_tag,
        channel=channel,
        exclude_prompt_id=exclude_prompt_id,
        include_parent_text=include_parent_text,
        top_k=top_k,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="inspect_client_supabase_attendance",
    description=(
        "QUANDO USAR: Sempre que o usuário pedir para investigar um atendimento real no Supabase do cliente, "
        "verificar a memória de conversa de um lead/morador (n8n_chat_histories ou Retell_calls_Mindflow), "
        "ou auditar quais ferramentas MCP e parâmetros foram executados em um atendimento (workflow_executions e workflow_step_executions). "
        "COMO USAR: Informe 'prompt_id' ou 'client_id'. Você pode filtrar por 'phone_or_session' (ex: '+5548996027108'), "
        "por 'execution_id' (UUID da execução EDW), ou usar 'query_mode' ('auto', 'latest_errors', 'latest') para buscar os últimos atendimentos do cliente."
    ),
)
async def mcp_inspect_client_supabase_attendance(
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    phone_or_session: Optional[str] = None,
    execution_id: Optional[str] = None,
    query_mode: str = "auto",
    limit: int = 20,
) -> str:
    res = await inspect_client_supabase_attendance_impl(
        prompt_id=prompt_id,
        client_id=client_id,
        phone_or_session=phone_or_session,
        execution_id=execution_id,
        query_mode=query_mode,
        limit=limit,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="get_arena_and_evals_context",
    description=(
        "QUANDO USAR: Sempre que o usuário pedir para ler a conversa atual do teste A/B (Arena A/B), "
        "comparar como a Versão A e a Versão B responderam no playground, ver os logs de tools acionadas na Arena, "
        "ou analisar os casos reprovados na última bateria de Evals da versão atual. "
        "COMO USAR: Chame com include_arena_chat=True e/ou include_eval_failures=True."
    ),
)
async def mcp_get_arena_and_evals_context(
    version_id: Optional[str] = None,
    prompt_id: Optional[int] = None,
    include_arena_chat: bool = True,
    include_eval_failures: bool = True,
) -> str:
    res = await get_arena_and_evals_context_impl(
        version_id=version_id,
        prompt_id=prompt_id,
        include_arena_chat=include_arena_chat,
        include_eval_failures=include_eval_failures,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="inspect_client_mcp_schemas",
    description=(
        "QUANDO USAR: Sempre que você precisar criar ou corrigir instruções de chamada de ferramentas MCP no prompt "
        "e quiser conferir os nomes exatos das tools e os parâmetros obrigatórios (JSON Schema) nos servidores MCP do cliente. "
        "COMO USAR: Passe o 'prompt_id' ou 'client_id' do prompt atual."
    ),
)
async def mcp_inspect_client_mcp_schemas(
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
) -> str:
    res = await inspect_client_mcp_schemas_impl(prompt_id=prompt_id, client_id=client_id)
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="read_mindflow_engineering_docs",
    description=(
        "QUANDO USAR: Sempre que precisar consultar as regras científicas detalhadas do Livro-Razão de Hipóteses "
        "(Teoria T-01 de Variáveis Simbólicas, H-04 protocolo '@', H-03 Re-hidratação Silenciosa, H-01 banida), "
        "a Anatomia Modular Canônica de 5 blocos ortogonais ou o Guia de Tabelas do Supabase. "
        "COMO USAR: Passe topic='prompt_hypotheses_ledger', 'canonical_prompt_anatomy', 'supabase_data_guide' ou 'all'."
    ),
)
async def mcp_read_mindflow_engineering_docs(topic: str = "all") -> str:
    res = read_mindflow_engineering_docs_impl(topic=topic)
    return json.dumps(res, ensure_ascii=False, indent=2)


# ============================================================================
# Definição OpenAI Function Calling + Dispatcher On-Demand para o Agente Gerador
# ============================================================================

def get_generator_openai_tools() -> List[Dict[str, Any]]:
    """Retorna os schemas OpenAI Function Calling com instruções ricas do MCP para o Agente Gerador."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_rag_prompt_blocks",
                "description": (
                    "QUANDO USAR: Acione sob demanda quando o usuário pedir para usar outro prompt como exemplo/referência, "
                    "ou quando você precisar consultar blocos modulares (<GlobalSettings>, <variaveis>, <persona>, <ferramentas_MCP>, "
                    "<few_shot_declarado>, técnicas de Treinamento SDR) no RAG Parent-Child de clientes ativos da MindFlow. "
                    "COMO USAR: Passe 'query' com os termos semânticos e opcionalmente 'section_tag' e 'channel'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Consulta semântica ou palavras-chave (ex: 'agendamento check_availability few_shot' ou 'contorno objeção SDR').",
                        },
                        "section_tag": {
                            "type": "string",
                            "description": "Filtro opcional de tag modular (ex: 'few_shot_declarado', 'ferramentas_MCP', 'variaveis', 'persona').",
                        },
                        "channel": {
                            "type": "string",
                            "description": "Filtro opcional de canal ('whatsapp', 'ligacao', 'fup', 'instagram').",
                        },
                        "include_parent_text": {
                            "type": "boolean",
                            "description": "Se True, retorna também o texto completo do prompt Pai além dos Child Chunks.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Quantidade máxima de blocos retornados (padrão: 4).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_client_supabase_attendance",
                "description": (
                    "QUANDO USAR: Acione sob demanda quando o usuário pedir para investigar um atendimento real no Supabase do cliente, "
                    "ler a memória de conversa de um telefone/sessão (n8n_chat_histories ou Retell_calls_Mindflow), "
                    "ou auditar o uso de tools e erros em uma execução (workflow_executions / workflow_step_executions). "
                    "COMO USAR: Passe 'phone_or_session' (ex: '+5548996027108'), 'execution_id' ou 'query_mode' ('auto', 'latest_errors', 'latest')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_or_session": {
                            "type": "string",
                            "description": "Número de telefone ou session_id do atendimento (ex: '+5548996027108').",
                        },
                        "execution_id": {
                            "type": "string",
                            "description": "UUID da execução na tabela workflow_executions, se disponível.",
                        },
                        "query_mode": {
                            "type": "string",
                            "enum": ["auto", "by_phone", "latest_errors", "latest"],
                            "description": "Modo de busca no Supabase do cliente.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Limite de mensagens de histórico a retornar (padrão: 20).",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_arena_and_evals_context",
                "description": (
                    "QUANDO USAR: Acione sob demanda quando o usuário pedir para ler a conversa atual na Arena A/B (teste A/B), "
                    "analisar como o bot respondeu no chat de teste, ou verificar os casos reprovados na última bateria de Evals. "
                    "COMO USAR: Defina include_arena_chat=True para ler a conversa A/B e include_eval_failures=True para ler falhas de evals e logs de tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_arena_chat": {
                            "type": "boolean",
                            "description": "Se True, retorna o histórico completo da conversa atual na Arena A/B (Versão A e Versão B).",
                        },
                        "include_eval_failures": {
                            "type": "boolean",
                            "description": "Se True, retorna os casos reprovados na última bateria de Evals e os logs de execução de tools.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_client_mcp_schemas",
                "description": (
                    "QUANDO USAR: Acione sob demanda quando precisar saber quais ferramentas MCP o cliente possui "
                    "e quais são os nomes exatos dos parâmetros (JSON Schema) aceitos por cada tool antes de escrever regras de MCP no prompt. "
                    "COMO USAR: Chame sem argumentos adicionais (usa automaticamente o prompt_id/client_id atual)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_mindflow_engineering_docs",
                "description": (
                    "QUANDO USAR: Acione sob demanda quando precisar consultar detalhes do Livro-Razão de Hipóteses "
                    "(T-01, H-04, H-03, H-01), a Anatomia Modular Canônica de 5 blocos ou o Guia de Dados do Supabase da MindFlow. "
                    "COMO USAR: Informe topic='prompt_hypotheses_ledger', 'canonical_prompt_anatomy', 'supabase_data_guide' ou 'all'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "enum": ["all", "prompt_hypotheses_ledger", "canonical_prompt_anatomy", "supabase_data_guide"],
                            "description": "Documento técnico a ser consultado.",
                        }
                    },
                },
            },
        },
    ]


TOOL_BADGE_LABELS: Dict[str, str] = {
    "search_rag_prompt_blocks": "📚 Buscou Referências no RAG Parent-Child",
    "inspect_client_supabase_attendance": "🗄️ Inspecionou Atendimento & Tools no Supabase",
    "get_arena_and_evals_context": "🔍 Consultou Conversa Arena A/B & Evals",
    "inspect_client_mcp_schemas": "🛠️ Inspecionou Schemas MCP do Cliente",
    "read_mindflow_engineering_docs": "📖 Consultou Docs de Engenharia & Hipóteses",
}


async def execute_generator_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executa uma tool on-demand solicitada pelo Agente Gerador e retorna
    tanto o payload para o LLM quanto o resumo amigável (badge) para a UI.
    """
    prompt_id = context.get("prompt_id")
    client_id = context.get("client_id")
    channel = context.get("channel")
    version_id = context.get("version_id")
    arena_history = context.get("arena_history")
    failing_examples = context.get("failing_examples")

    label = TOOL_BADGE_LABELS.get(tool_name, f"⚡ Tool: {tool_name}")

    if tool_name == "search_rag_prompt_blocks":
        res = await search_rag_prompt_blocks_impl(
            query=arguments.get("query", ""),
            section_tag=arguments.get("section_tag"),
            channel=arguments.get("channel") or None,
            exclude_prompt_id=prompt_id,
            include_parent_text=bool(arguments.get("include_parent_text", False)),
            top_k=int(arguments.get("top_k", 4)),
        )
        summary = f"{res.get('total_results', 0)} bloco(s) recuperado(s) para '{arguments.get('query', '')[:35]}'"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "inspect_client_supabase_attendance":
        res = await inspect_client_supabase_attendance_impl(
            prompt_id=prompt_id,
            client_id=client_id,
            phone_or_session=arguments.get("phone_or_session"),
            execution_id=arguments.get("execution_id"),
            query_mode=arguments.get("query_mode", "auto"),
            limit=int(arguments.get("limit", 20)),
        )
        n_msgs = len(res.get("chat_memory", [])) + len(res.get("voice_calls", []))
        n_steps = len(res.get("workflow_steps", []))
        target_desc = arguments.get("phone_or_session") or arguments.get("execution_id") or arguments.get("query_mode", "recente")
        summary = f"Filtro '{target_desc}': {n_msgs} msg(s) e {n_steps} step(s) de tools"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "get_arena_and_evals_context":
        res = await get_arena_and_evals_context_impl(
            arena_history=arena_history,
            failing_examples=failing_examples,
            version_id=version_id,
            prompt_id=prompt_id,
            include_arena_chat=bool(arguments.get("include_arena_chat", True)),
            include_eval_failures=bool(arguments.get("include_eval_failures", True)),
        )
        ab = res.get("arena_conversation_ab") or {}
        turns_a = len(ab.get("A") or [])
        turns_b = len(ab.get("B") or [])
        n_fails = len(res.get("recent_eval_failures") or [])
        summary = f"Arena ({turns_a} msgs A / {turns_b} msgs B) · {n_fails} falha(s) em Evals"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "inspect_client_mcp_schemas":
        res = await inspect_client_mcp_schemas_impl(prompt_id=prompt_id, client_id=client_id)
        summary = f"{res.get('total_tools', 0)} tool(s) MCP inspecionada(s)"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "read_mindflow_engineering_docs":
        topic = arguments.get("topic", "all")
        res = read_mindflow_engineering_docs_impl(topic=topic)
        summary = f"Tópico consultado: {topic}"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    return {
        "output": {"error": f"Tool '{tool_name}' desconhecida."},
        "badge": {"tool_name": tool_name, "label": label, "summary": "Tool não encontrada"},
    }
