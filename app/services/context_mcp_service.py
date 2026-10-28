import json
import re
from uuid import UUID
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from supabase import create_client
from app.database import async_session_maker, get_production_supabase
from app.models import (
    EvalDataset,
    EvalRun,
    EvalCaseResult,
    PromptVersion,
    ToolExecutionLog,
    PromptEngineeringDoc,
    get_utc_now,
)
from app.services.rag_service import search_rag_chunks
from app.services.mcp_service import get_client_mcp_urls, get_client_tools_and_map
from app.services.eval_service import evaluate_single_turn_case

# Instância oficial FastMCP expondo instruções ricas e detalhadas para Agentes
prompt_ops_mcp = FastMCP(
    "prompt_ops_context_mcp",
    instructions=(
        "Servidor MCP oficial de Contexto, RAG Parent-Child, Sincronização de Hipóteses e Diagnóstico Multi-Tenant da MindFlow (PromptOps). "
        "Compartilhado em tempo real entre o Agente Antigravity (IDE) e o Agente Copilot do PromptOps (Web UI). "
        "Utilize estas ferramentas SOB DEMANDA (on-demand) sempre que precisar consultar ou atualizar: "
        "(1) Exemplos e blocos modulares de outros prompts de clientes ativos ou de Treinamento SDR no RAG Parent-Child; "
        "(2) Memória conversacional (n8n_chat_histories / Retell) e rastreio de execução de tools (workflow_step_executions) "
        "de um atendimento específico no Supabase isolado do cliente; "
        "(3) A conversa atual que está ocorrendo na Arena A/B ou falhas recentes nos Evals; "
        "(4) Os schemas exatos das ferramentas MCP habilitadas para o cliente; "
        "(5) Ler ou atualizar a Documentação Viva de Engenharia de Prompts (Livro-Razão de Hipóteses prompt_hypotheses_ledger, "
        "Memória Ativa active_memory_xml, Anatomia Canônica e Guia Supabase)."
    ),
)


ENGINEERING_DOCS_TITLES: Dict[str, str] = {
    "prompt_hypotheses_ledger": "Livro-Razão de Hipóteses Científicas de Prompts (Prompt Hypotheses Ledger)",
    "active_memory_xml": "Memória Ativa do Compilador de Prompts Corporativos (active_memory.xml)",
    "canonical_prompt_anatomy": "Anatomia Modular Canônica do Prompt (5 Blocos Ortogonais)",
    "supabase_data_guide": "Guia de Estrutura de Dados Multi-Tenant (Supabase Master & Cliente)",
}

ENGINEERING_DOCS_LIBRARY: Dict[str, str] = {
    "prompt_hypotheses_ledger": """# Livro-Razão de Hipóteses Científicas de Prompts (Prompt Hypotheses Ledger — M-SPEM v2)

> **Documento Oficial Sincronizado (Antigravity <-> PromptOps):** Livro-Razão Vivo de Hipóteses Gerais (`G-*`), Modos de Falha (`FM-*`) e Replicações Independentes da MindFlow, derivado deterministicamente por `prompt_science/score.py`.

## 1. Arquitetura em 3 Níveis e Regra de Ouro (`1 Rodada = 1 Mutação ΔP = 1 Hipótese G-*`)
- **Nível 0 (Especificação do Cliente):** Dados factuais do tenant (telefones, horários, preços) vivem no prompt do cliente + Eval de regressão, nunca como hipótese.
- **Nível 1 (Instância Local):** Aplicação parametrizada em um agente/API vinculada a uma `G-*`.
- **Nível 2 (Hipótese Geral `G-*`):** Princípios cognitivos/estruturais transferíveis entre domínios e clientes, organizados por Modo de Falha (`FM-01` a `FM-07`).

## 2. Regras de Pontuação por Replicação Independente (Salvaguardas S1–S9)
- **Unidade de Ponto = 1 Replicação Pareada Independente** (Controle $v_N$ vs. Tratamento $v_N + \\Delta P$, com pré-registro `sha256`, na temperatura de produção, critérios positivos e negativos + $\\ge 20\\%$ de anti-casos `is_anti_case=true`).
- **Aprovação em Replicação Pareada (`+1 ponto`):** Superioridade pareada ($P(\\Delta > 0) \\ge 0.90$ ou McNemar $p \\le 0.10$), aprovação $\\ge 90\\%$, zero Contaminação Tipo A/B e zero regressão em anti-casos.
- **Falha em Replicação Pareada (`-2 pontos`):** Contaminação Tipo A/B, regressão em anti-casos ou desempenho inferior ao Controle.
- **Promoção a Teoria (`🏆 TEORIA CONSOLIDADA`):** Exige **`>= 8 pontos`** acumulados em **`>= 3 domínios/agentes distintos`** e **`>= 2 famílias de modelos`**.
- **Refutação de Escopo (`<= -4 pontos`):** Declarada **`❌ X REFUTADA NO ESCOPO`**, bloqueando o padrão no escopo onde o dano foi replicado.

## 3. Auditoria Epistemológica e Migração Histórica (`T-01` -> `G-SYMB` + `G-FEWSHOT-STRUCT` + `G-COD`)
- **Auditoria do legado `T-01` (`H-02`):** O antigo pacote composto `T-01` foi desmembrado nas hipóteses gerais ortogonais **`G-SYMB`**, **`G-FEWSHOT-STRUCT`** e **`G-COD`**, recontando honestamente por replicação independente (`+1 pt` em `kravi_sac_whatsapp`) após correção do vazamento simbólico Tipo B com **`G-PRECOND`**.

## 4. Ranking de Hipóteses Gerais em Teste (`G-*`)
- **1º `G-SYMB` (`+1 pt`, 1/3 domínios — Legado `T-01`/`H-02`):** Isolamento de Argumentos Dinâmicos por Ponteiros Simbólicos Tipados (`[var]`).
- **2º `G-PRIOR` (`+1 pt`, 1/3 domínios — Legado `H-04`):** Override de Prior de Pré-Treinamento por Contraste Sintático Negativo + Consequência de Erro (ex: `'@'` vs `null`/`""`).
- **3º `G-PRECOND` (`+1 pt`, 1/3 domínios — Legado `H-03`):** Cláusula de Pré-Condição e Re-Hidratação Silenciosa quando Ponteiro Simbólico é Nulo.
- **4º `G-FEWSHOT-STRUCT` (`+1 pt`, 1/3 domínios — Legado `T-01`/`H-02`):** Few-Shot de Transição de Estado Explícita (`<estado_sessao_anterior>` -> `<raciocinio_interno_cod>` -> `<chamada_ferramenta>` -> `<estado_sessao_atualizado>`).
- **5º `G-FALLBACK` (`+1 pt`, 1/3 domínios — Legado `H-09`):** Cadeia Hierárquica Univariada Estrita de Identificação (Um Identificador por Turno).
- **6º `G-AMBIG` (`+1 pt`, 1/3 domínios — Legado `H-09`):** Escalonamento Determinístico quando Busca Retorna Múltiplos Candidatos sem Chave de Desambiguação.
- **7º `G-COD` (`0 pts`, 0/3 domínios — Legado `T-01`/`H-02`):** Chain of Draft (CoD) Conectado à Decisão de Ferramenta e Estado.
- **8º `G-ASYM` (`0 pts`, 0/3 domínios — Legado `H-08`):** Regra de Decisão Assimétrica sob Alto Custo de Falso Negativo Comercial.

## 5. Hipóteses Refutadas no Escopo (`<= -4 pontos` — Bloqueio de Escopo)
- **`G-LIT-ID` (`-4 pontos` — Legado `H-01` — ❌ X REFUTADA NO ESCOPO):** Exemplos Few-Shot com IDs Primários/Relacionais Literais em Fluxos Multi-Tenant Dinâmicos. Causa Contaminação Tipo A por anchoring bias.""",
    "active_memory_xml": """<memoria_ativa versao="2.0" escopo="prompts_de_clientes_tabela_prompts" gerada_por="prompt_science/score.py">
  <regras_de_geracao>
    <regra tipo="IMUTAVEL" id="R001">Todo prompt de cliente deve declarar explicitamente persona, contexto, formato de output e critério de encerramento.</regra>
    <regra tipo="IMUTAVEL" id="R002">Estrutura de output sempre em XML ou Markdown conforme padrão da empresa. Nunca misturar os dois no mesmo bloco.</regra>
    <regra tipo="IMUTAVEL" id="R003_T01" escopo_refutado="G-LIT-ID" origem="docs/prompt_hypotheses_ledger.md#G-LIT-ID">Proibição de IDs Literais de Sessão (Escopo Refutado G-LIT-ID / H-01; decomposto em G-SYMB + G-FEWSHOT-STRUCT + G-COD): Em prompts com tool calling que manipulam chaves dinâmicas de sessão/tenant, é proibido inserir IDs numéricos literais de casos antigos em exemplos ou regras.</regra>
    <regra tipo="MUTAVEL" id="R004_G_SYMB" status_cientifico="em_teste_score_1pt" legado="T-01/H-02" ultima_atualizacao="2026-10-28">[G-SYMB | +1 pt] Isolamento de Argumentos Dinâmicos por Ponteiros Simbólicos Tipados ([var]) em bloco &lt;variaveis&gt;.</regra>
    <regra tipo="MUTAVEL" id="R005_G_PRIOR" status_cientifico="em_teste_score_1pt" legado="H-04" ultima_atualizacao="2026-10-28">[G-PRIOR | +1 pt] Override de Prior de Pré-Treinamento por Contraste Sintático Negativo + Consequência de Erro (ex: '@' em vez de null/"").</regra>
    <regra tipo="MUTAVEL" id="R006_G_PRECOND" status_cientifico="em_teste_score_1pt" legado="H-03" ultima_atualizacao="2026-10-28">[G-PRECOND | +1 pt] Cláusula de Pré-Condição e Re-Hidratação Silenciosa quando Ponteiro Simbólico é Nulo.</regra>
  </regras_de_geracao>
  <erros_registrados>
    <erro id="E001" status="aberto" data="2026-10-20" ref_hipotese="G-LIT-ID/H-01">Inserir exemplos reais de casos antigos com IDs literais (Contaminação Tipo A).</erro>
    <erro id="E002" status="aberto" data="2026-10-20" ref_hipotese="G-PRECOND/REF-01">Enviar o nome textual do placeholder ("singleAccountId") como argumento da tool quando o histórico está truncado (Contaminação Tipo B).</erro>
    <erro id="E003" status="resolvido" data="2026-10-28" ref_hipotese="AUDITORIA-M-SPEM-V2">Avaliar casos apenas pela ausência de IDs proibidos sem verificar resolução positiva de argumentos e anti-casos.</erro>
  </erros_registrados>
</memoria_ativa>""",
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


async def _ensure_engineering_docs_seeded(session) -> None:
    """Garante que os documentos base existam na tabela prompt_engineering_docs."""
    res = await session.execute(select(PromptEngineeringDoc))
    existing = {row.doc_key: row for row in res.scalars().all()}
    changed = False
    for k, default_text in ENGINEERING_DOCS_LIBRARY.items():
        if k not in existing:
            doc = PromptEngineeringDoc(
                doc_key=k,
                title=ENGINEERING_DOCS_TITLES.get(k, k),
                content=default_text,
                version=1,
                updated_by="system_seed",
                change_summary="Seed inicial da documentação compartilhada Antigravity <-> PromptOps",
            )
            session.add(doc)
            changed = True
    if changed:
        await session.commit()


async def read_mindflow_engineering_docs_impl(
    topic: str = "all",
    _session_factory: Any = None,
) -> Dict[str, Any]:
    """
    Retorna a documentação oficial viva de Engenharia de Prompts e Hipóteses da MindFlow
    persistida na tabela compartilhada prompt_engineering_docs (com fallback em memória).
    """
    t = (topic or "all").strip().lower()
    factory = _session_factory or async_session_maker
    try:
        async with factory() as session:
            await _ensure_engineering_docs_seeded(session)
            if t != "all":
                res = await session.execute(
                    select(PromptEngineeringDoc).where(PromptEngineeringDoc.doc_key == t)
                )
                doc = res.scalar_one_or_none()
                if doc:
                    return {
                        "topic": doc.doc_key,
                        "title": doc.title,
                        "content": doc.content,
                        "version": doc.version,
                        "updated_by": doc.updated_by,
                        "change_summary": doc.change_summary,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                    }
            res_all = await session.execute(select(PromptEngineeringDoc))
            docs_all = res_all.scalars().all()
            return {
                "topic": "all",
                "documents": {
                    d.doc_key: {
                        "title": d.title,
                        "content": d.content,
                        "version": d.version,
                        "updated_by": d.updated_by,
                        "change_summary": d.change_summary,
                        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
                    }
                    for d in docs_all
                },
            }
    except Exception as e:
        print(f"[Context MCP Warning] Fallback in-memory para read_mindflow_engineering_docs: {e}")
        if t in ENGINEERING_DOCS_LIBRARY:
            return {"topic": t, "content": ENGINEERING_DOCS_LIBRARY[t], "version": 1, "updated_by": "fallback"}
        return {"topic": "all", "documents": ENGINEERING_DOCS_LIBRARY}


async def update_mindflow_engineering_doc_impl(
    doc_key: str,
    content: str,
    updated_by: str = "prompt_ops_copilot",
    change_summary: str = "Atualização via agente",
    title: Optional[str] = None,
    _session_factory: Any = None,
) -> Dict[str, Any]:
    """
    Atualiza de forma transacional e versionada um documento de engenharia/hipóteses
    compartilhado entre o Agente Antigravity e o Agente Copilot do PromptOps.
    """
    key = (doc_key or "").strip().lower()
    if not key:
        raise ValueError("doc_key é obrigatório.")
    factory = _session_factory or async_session_maker
    async with factory() as session:
        await _ensure_engineering_docs_seeded(session)
        res = await session.execute(
            select(PromptEngineeringDoc).where(PromptEngineeringDoc.doc_key == key)
        )
        doc = res.scalar_one_or_none()
        if doc:
            doc.content = content
            doc.version = (doc.version or 1) + 1
            doc.updated_by = updated_by
            doc.change_summary = change_summary
            if title:
                doc.title = title
        else:
            doc = PromptEngineeringDoc(
                doc_key=key,
                title=title or ENGINEERING_DOCS_TITLES.get(key, key),
                content=content,
                version=1,
                updated_by=updated_by,
                change_summary=change_summary,
            )
            session.add(doc)
        await session.commit()
        await session.refresh(doc)
        # Mantém cache em memória sincronizado
        ENGINEERING_DOCS_LIBRARY[key] = doc.content
        return {
            "doc_key": doc.doc_key,
            "title": doc.title,
            "content": doc.content,
            "version": doc.version,
            "updated_by": doc.updated_by,
            "change_summary": doc.change_summary,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }


def _serialize_eval_case(c: EvalDataset) -> Dict[str, Any]:
    return {
        "id": str(c.id),
        "prompt_id": c.prompt_id,
        "client_id": c.client_id,
        "channel": c.channel,
        "scenario_name": c.scenario_name,
        "test_type": c.test_type,
        "input_data": c.input_data,
        "expected_criteria": c.expected_criteria,
        "requires_tool": bool(c.requires_tool),
        "expected_tool_name": c.expected_tool_name,
        "is_refutation_case": bool(getattr(c, "is_refutation_case", False)),
        "is_anti_case": bool(getattr(c, "is_anti_case", False)),
        "forbid_tool_call": bool(getattr(c, "forbid_tool_call", False)),
        "expected_tool_args": getattr(c, "expected_tool_args", None) or {},
        "forbidden_literal_ids": getattr(c, "forbidden_literal_ids", None) or [],
        "forbidden_symbolic_placeholders": getattr(c, "forbidden_symbolic_placeholders", None) or [],
        "created_by": getattr(c, "created_by", None),
        "updated_by": getattr(c, "updated_by", None),
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


async def list_prompt_eval_datasets_impl(
    prompt_id: Optional[int] = None,
    channel: Optional[str] = None,
    _session_factory: Any = None,
) -> Dict[str, Any]:
    """Lista todos os casos de teste ativos (incluindo Casos de Refutação Popperianos e Anti-Casos S5) de um prompt/canal."""
    factory = _session_factory or async_session_maker
    async with factory() as session:
        stmt = select(EvalDataset).where(EvalDataset.is_active == True).order_by(desc(EvalDataset.created_at))
        if prompt_id is not None:
            stmt = stmt.where(EvalDataset.prompt_id == int(prompt_id))
        if channel:
            stmt = stmt.where(EvalDataset.channel == channel)
        res = await session.execute(stmt)
        rows = res.scalars().all()
        cases = [_serialize_eval_case(r) for r in rows]
        return {
            "prompt_id": prompt_id,
            "channel": channel,
            "total_cases": len(cases),
            "refutation_cases_count": sum(1 for x in cases if x.get("is_refutation_case")),
            "anti_cases_count": sum(1 for x in cases if x.get("is_anti_case")),
            "cases": cases,
        }


async def manage_prompt_eval_case_impl(
    action: str,
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    channel: str = "whatsapp",
    case_id: Optional[str] = None,
    scenario_name: Optional[str] = None,
    user_message: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None,
    expected_criteria: Optional[str] = None,
    requires_tool: bool = False,
    expected_tool_name: Optional[str] = None,
    is_refutation_case: bool = False,
    is_anti_case: bool = False,
    forbid_tool_call: bool = False,
    expected_tool_args: Optional[Dict[str, Any]] = None,
    forbidden_literal_ids: Optional[List[str]] = None,
    forbidden_symbolic_placeholders: Optional[List[str]] = None,
    author_email: str = "prompt_ops_copilot",
    _session_factory: Any = None,
) -> Dict[str, Any]:
    """
    Cria, edita ou exclui (soft-delete) um caso de teste de Evals seguindo o protocolo científico de Prompt Science (M-SPEM v2):
    suporta Casos de Refutação Popperianos (is_refutation_case=True), Anti-Casos S5 (is_anti_case=True, forbid_tool_call=True),
    critérios positivos de argumentos (expected_tool_args) e guardrails anti-contaminação Tipo A/B.
    """
    act = (action or "create").strip().lower()
    factory = _session_factory or async_session_maker

    async with factory() as session:
        if act == "create":
            if prompt_id is None:
                raise ValueError("prompt_id é obrigatório para criar um caso de Eval.")
            final_input = input_data if input_data is not None else {"user_message": user_message or ""}
            item = EvalDataset(
                prompt_id=int(prompt_id),
                client_id=str(client_id or prompt_id),
                channel=channel or "whatsapp",
                scenario_name=scenario_name or "Caso de Teste Científico",
                test_type="single_turn",
                input_data=final_input,
                expected_criteria=expected_criteria or "",
                requires_tool=bool(requires_tool),
                expected_tool_name=expected_tool_name,
                is_refutation_case=bool(is_refutation_case),
                is_anti_case=bool(is_anti_case),
                forbid_tool_call=bool(forbid_tool_call),
                expected_tool_args=expected_tool_args or {},
                forbidden_literal_ids=forbidden_literal_ids or [],
                forbidden_symbolic_placeholders=forbidden_symbolic_placeholders or [],
                created_by=author_email,
                is_active=True,
                created_at=get_utc_now(),
                updated_at=get_utc_now(),
            )
            session.add(item)
            await session.commit()
            await session.refresh(item)
            return {"action": "created", "case": _serialize_eval_case(item)}

        if not case_id:
            raise ValueError("case_id é obrigatório para update ou delete.")
        cid = UUID(str(case_id)) if not isinstance(case_id, UUID) else case_id
        res = await session.execute(select(EvalDataset).where(EvalDataset.id == cid))
        item = res.scalar_one_or_none()
        if not item:
            raise ValueError(f"Caso de teste {case_id} não encontrado.")

        if act == "update":
            if scenario_name is not None:
                item.scenario_name = scenario_name
            if user_message is not None:
                curr = dict(item.input_data or {})
                curr["user_message"] = user_message
                item.input_data = curr
            elif input_data is not None:
                item.input_data = input_data
            if expected_criteria is not None:
                item.expected_criteria = expected_criteria
            if requires_tool is not None:
                item.requires_tool = bool(requires_tool)
            if expected_tool_name is not None:
                item.expected_tool_name = expected_tool_name
            if is_refutation_case is not None:
                item.is_refutation_case = bool(is_refutation_case)
            if is_anti_case is not None:
                item.is_anti_case = bool(is_anti_case)
            if forbid_tool_call is not None:
                item.forbid_tool_call = bool(forbid_tool_call)
            if expected_tool_args is not None:
                item.expected_tool_args = expected_tool_args
            if forbidden_literal_ids is not None:
                item.forbidden_literal_ids = forbidden_literal_ids
            if forbidden_symbolic_placeholders is not None:
                item.forbidden_symbolic_placeholders = forbidden_symbolic_placeholders
            item.updated_by = author_email
            item.updated_at = get_utc_now()
            await session.commit()
            await session.refresh(item)
            return {"action": "updated", "case": _serialize_eval_case(item)}

        if act == "delete":
            item.is_active = False
            item.deleted_by = author_email
            item.deleted_at = get_utc_now()
            await session.commit()
            return {"action": "deleted", "case_id": str(case_id), "deleted_by": author_email}

        raise ValueError(f"Ação '{action}' inválida. Use 'create', 'update' ou 'delete'.")


async def run_prompt_eval_suite_impl(
    version_id: Optional[Any] = None,
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    channel: Optional[str] = "whatsapp",
    live_prompt_text: Optional[str] = None,
    test_live_editor_prompt: bool = True,
    _session_factory: Any = None,
) -> Dict[str, Any]:
    """
    Executa a bateria completa de Evals (com chamadas MCP reais, LLM Judge e Detector de Contaminação Tipo A/B).
    Se uma version_id estiver selecionada, registra a EvalRun no banco e atualiza o eval_score da versão
    (liberando o Quality Gate >= 90%). Se test_live_editor_prompt=True e live_prompt_text for fornecido,
    avalia o texto em edição ao vivo.
    """
    factory = _session_factory or async_session_maker
    async with factory() as session:
        version_obj = None
        if version_id:
            try:
                vid = UUID(str(version_id)) if not isinstance(version_id, UUID) else version_id
                res_v = await session.execute(select(PromptVersion).where(PromptVersion.id == vid))
                version_obj = res_v.scalar_one_or_none()
            except Exception:
                version_obj = None

        eff_prompt_id = int(prompt_id or (version_obj.prompt_id if version_obj else 0))
        eff_client_id = str(client_id or (version_obj.client_id if version_obj else eff_prompt_id))
        eff_channel = channel or (version_obj.channel if version_obj else "whatsapp")
        eff_model = version_obj.model_name if version_obj else "gpt-4.1"
        eff_temp = version_obj.temperature if version_obj else 0.7
        eff_text = (
            live_prompt_text
            if (test_live_editor_prompt and live_prompt_text)
            else (version_obj.prompt_text if version_obj else (live_prompt_text or ""))
        )

        stmt_d = select(EvalDataset).where(
            EvalDataset.prompt_id == eff_prompt_id,
            EvalDataset.is_active == True,
        )
        if eff_channel:
            stmt_d = stmt_d.where(EvalDataset.channel == eff_channel)
        res_d = await session.execute(stmt_d)
        cases = res_d.scalars().all()

        if not cases:
            stmt_fb = select(EvalDataset).where(
                EvalDataset.prompt_id == eff_prompt_id,
                EvalDataset.is_active == True,
            )
            res_fb = await session.execute(stmt_fb)
            cases = res_fb.scalars().all()

        if not cases:
            return {
                "error": "Nenhum caso de teste ativo cadastrado para este prompt. Crie ao menos 1 caso com manage_prompt_eval_case antes de rodar.",
                "total_cases": 0,
                "passed_cases": 0,
                "approval_rate": 0.0,
                "quality_gate_passed": False,
            }

        run_obj = None
        if version_obj:
            # Se o texto ao vivo do editor foi testado e difere da versão draft, sincroniza na versão draft não-prod
            if test_live_editor_prompt and live_prompt_text and not version_obj.version_label.startswith("prod-"):
                version_obj.prompt_text = live_prompt_text
            run_obj = EvalRun(
                version_id=version_obj.id,
                prompt_id=eff_prompt_id,
                status="running",
                total_cases=len(cases),
                passed_cases=0,
                approval_rate=0.0,
                triggered_by="copilot_agent",
                started_at=get_utc_now(),
            )
            session.add(run_obj)
            await session.commit()
            await session.refresh(run_obj)

        passed_count = 0
        detailed_results = []

        for c in cases:
            ev = await evaluate_single_turn_case(
                prompt_text=eff_text,
                model_name=eff_model,
                temperature=eff_temp,
                case_input=c.input_data,
                expected_criteria=c.expected_criteria,
                prompt_id=eff_prompt_id,
                client_id=eff_client_id,
                requires_tool=bool(c.requires_tool),
                expected_tool_name=c.expected_tool_name,
                version_id=version_obj.id if version_obj else None,
                forbidden_literal_ids=getattr(c, "forbidden_literal_ids", None) or [],
                forbidden_symbolic_placeholders=getattr(c, "forbidden_symbolic_placeholders", None) or [],
                forbid_tool_call=bool(getattr(c, "forbid_tool_call", False)),
                expected_tool_args=getattr(c, "expected_tool_args", None) or {},
            )
            if ev["passed"]:
                passed_count += 1

            if run_obj:
                case_row = EvalCaseResult(
                    run_id=run_obj.id,
                    dataset_id=c.id,
                    passed=ev["passed"],
                    generated_response=ev["generated_response"],
                    judge_reasoning=ev["judge_reasoning"],
                    tool_called=ev.get("tool_called"),
                    tool_call_valid=ev.get("tool_call_valid"),
                    execution_time_ms=ev["execution_time_ms"],
                )
                session.add(case_row)

            detailed_results.append({
                "dataset_id": str(c.id),
                "scenario_name": c.scenario_name,
                "is_refutation_case": bool(getattr(c, "is_refutation_case", False)),
                "is_anti_case": bool(getattr(c, "is_anti_case", False)),
                "passed": ev["passed"],
                "is_contaminated": ev.get("is_contaminated", False),
                "contamination_reason": ev.get("contamination_reason"),
                "generated_response": ev["generated_response"],
                "judge_reasoning": ev["judge_reasoning"],
                "tool_called": ev.get("tool_called"),
                "tool_call_valid": ev.get("tool_call_valid"),
                "execution_time_ms": ev["execution_time_ms"],
            })

        approval_rate = passed_count / len(cases)
        if run_obj and version_obj:
            run_obj.passed_cases = passed_count
            run_obj.approval_rate = approval_rate
            run_obj.status = "completed"
            run_obj.completed_at = get_utc_now()
            version_obj.eval_score = approval_rate
            if approval_rate >= 0.90 and not version_obj.version_label.startswith("prod-"):
                version_obj.status = "approved"
            await session.commit()

        return {
            "run_id": str(run_obj.id) if run_obj else None,
            "version_id": str(version_obj.id) if version_obj else None,
            "version_label": version_obj.version_label if version_obj else "live_editor",
            "total_cases": len(cases),
            "passed_cases": passed_count,
            "failed_cases": len(cases) - passed_count,
            "approval_rate": approval_rate,
            "approval_pct": round(approval_rate * 100),
            "quality_gate_passed": approval_rate >= 0.90,
            "cases": detailed_results,
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
        "a Memória Ativa (<memoria_ativa>), a Anatomia Modular Canônica de 5 blocos ortogonais ou o Guia de Tabelas do Supabase. "
        "COMO USAR: Passe topic='prompt_hypotheses_ledger', 'active_memory_xml', 'canonical_prompt_anatomy', 'supabase_data_guide' ou 'all'."
    ),
)
async def mcp_read_mindflow_engineering_docs(topic: str = "all") -> str:
    res = await read_mindflow_engineering_docs_impl(topic=topic)
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="update_mindflow_engineering_doc",
    description=(
        "QUANDO USAR: Sempre que uma nova hipótese científica for criada/testada/pontuada/refutada no Livro-Razão "
        "(doc_key='prompt_hypotheses_ledger') ou uma nova regra/erro for adicionado à memória ativa (doc_key='active_memory_xml'). "
        "Qualquer atualização feita aqui fica imediatamente visível tanto para o Agente do Antigravity quanto para o Agente do PromptOps. "
        "COMO USAR: Informe 'doc_key', 'content' (texto completo atualizado do documento), 'updated_by' e 'change_summary'."
    ),
)
async def mcp_update_mindflow_engineering_doc(
    doc_key: str,
    content: str,
    updated_by: str = "mcp_agent",
    change_summary: str = "Atualização sincronizada via MCP",
) -> str:
    res = await update_mindflow_engineering_doc_impl(
        doc_key=doc_key,
        content=content,
        updated_by=updated_by,
        change_summary=change_summary,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="list_prompt_eval_datasets",
    description=(
        "QUANDO USAR: Sempre que precisar listar os casos de teste e Casos de Refutação Científica (Evals) "
        "cadastrados para um prompt e canal antes de criar novos casos ou rodar a bateria."
    ),
)
async def mcp_list_prompt_eval_datasets(
    prompt_id: int,
    channel: Optional[str] = None,
) -> str:
    res = await list_prompt_eval_datasets_impl(prompt_id=prompt_id, channel=channel)
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="manage_prompt_eval_case",
    description=(
        "QUANDO USAR: Sempre que precisar criar ('create'), editar ('update') ou excluir ('delete') um caso de teste, "
        "Caso de Refutação Científica Popperiano ou Anti-Caso S5 nos Evals do prompt. "
        "REGRA CIENTÍFICA (M-SPEM v2 / prompt_engineer.md): Ao testar uma hipótese ou corrigir um bug de atendimento/tools, "
        "defina is_refutation_case=True, preencha forbidden_literal_ids (Contaminação Tipo A), "
        "forbidden_symbolic_placeholders (Contaminação Tipo B) e expected_tool_args (critério positivo), "
        "além de criar >= 20% de anti-casos (is_anti_case=True, forbid_tool_call=True)."
    ),
)
async def mcp_manage_prompt_eval_case(
    action: str,
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    channel: str = "whatsapp",
    case_id: Optional[str] = None,
    scenario_name: Optional[str] = None,
    user_message: Optional[str] = None,
    expected_criteria: Optional[str] = None,
    requires_tool: bool = False,
    expected_tool_name: Optional[str] = None,
    is_refutation_case: bool = False,
    is_anti_case: bool = False,
    forbid_tool_call: bool = False,
    expected_tool_args: Optional[Dict[str, Any]] = None,
    forbidden_literal_ids: Optional[List[str]] = None,
    forbidden_symbolic_placeholders: Optional[List[str]] = None,
) -> str:
    res = await manage_prompt_eval_case_impl(
        action=action,
        prompt_id=prompt_id,
        client_id=client_id,
        channel=channel,
        case_id=case_id,
        scenario_name=scenario_name,
        user_message=user_message,
        expected_criteria=expected_criteria,
        requires_tool=requires_tool,
        expected_tool_name=expected_tool_name,
        is_refutation_case=is_refutation_case,
        is_anti_case=is_anti_case,
        forbid_tool_call=forbid_tool_call,
        expected_tool_args=expected_tool_args,
        forbidden_literal_ids=forbidden_literal_ids,
        forbidden_symbolic_placeholders=forbidden_symbolic_placeholders,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@prompt_ops_mcp.tool(
    name="run_prompt_eval_suite",
    description=(
        "QUANDO USAR: Sempre que o usuário pedir para rodar os Evals da versão atual ou quando você quiser validar "
        "cientificamente as alterações aplicadas no prompt contra os casos de teste e refutação (Quality Gate >= 90%). "
        "Executa as chamadas MCP reais, o Detector de Contaminação Tipo A/B e o LLM Juiz, atualizando o eval_score da versão."
    ),
)
async def mcp_run_prompt_eval_suite(
    version_id: Optional[str] = None,
    prompt_id: Optional[int] = None,
    channel: str = "whatsapp",
    live_prompt_text: Optional[str] = None,
    test_live_editor_prompt: bool = True,
) -> str:
    res = await run_prompt_eval_suite_impl(
        version_id=version_id,
        prompt_id=prompt_id,
        channel=channel,
        live_prompt_text=live_prompt_text,
        test_live_editor_prompt=test_live_editor_prompt,
    )
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
                    "QUANDO USAR: Acione sob demanda quando precisar consultar detalhes sincronizados do Livro-Razão de Hipóteses "
                    "(T-01, H-04, H-03, H-01 e novas hipóteses gravadas pelo Antigravity), Memória Ativa (<memoria_ativa>), "
                    "Anatomia Modular Canônica de 5 blocos ou Guia de Dados do Supabase da MindFlow. "
                    "COMO USAR: Informe topic='prompt_hypotheses_ledger', 'active_memory_xml', 'canonical_prompt_anatomy', 'supabase_data_guide' ou 'all'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "enum": ["all", "prompt_hypotheses_ledger", "active_memory_xml", "canonical_prompt_anatomy", "supabase_data_guide"],
                            "description": "Documento técnico a ser consultado.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_mindflow_engineering_doc",
                "description": (
                    "QUANDO USAR: Acione quando o usuário pedir para registrar, atualizar ou pontuar uma hipótese científica "
                    "no Livro-Razão (doc_key='prompt_hypotheses_ledger') ou atualizar uma regra/erro na memória ativa "
                    "(doc_key='active_memory_xml') para que fique sincronizado com o Agente do Antigravity. "
                    "Importante: Leia antes com read_mindflow_engineering_docs para preservar o conteúdo existente ao adicionar itens."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_key": {
                            "type": "string",
                            "enum": ["prompt_hypotheses_ledger", "active_memory_xml", "canonical_prompt_anatomy", "supabase_data_guide"],
                            "description": "Chave do documento compartilhado a ser atualizado.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Conteúdo completo atualizado do documento.",
                        },
                        "change_summary": {
                            "type": "string",
                            "description": "Resumo curto do que foi adicionado/alterado.",
                        },
                    },
                    "required": ["doc_key", "content", "change_summary"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_prompt_eval_datasets",
                "description": (
                    "QUANDO USAR: Acione quando precisar listar todos os casos de teste de Evals e Casos de Refutação Científica "
                    "ativos cadastrados para o prompt/canal atual."
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
                "name": "manage_prompt_eval_case",
                "description": (
                    "QUANDO USAR: Acione sempre que o usuário pedir para criar, editar ou excluir um caso de teste de Eval, "
                    "ou quando você criar um Caso de Refutação Científica Popperiano / Anti-Caso S5 (prompt_engineer.md) para testar uma hipótese/correção. "
                    "COMO USAR: Passe action='create' (ou 'update'/'delete'), scenario_name, user_message, expected_criteria (rubrica rigorosa para o LLM Judge), "
                    "requires_tool, expected_tool_name, is_refutation_case=True (se for teste de refutação), is_anti_case=True e forbid_tool_call=True (se for anti-caso S5), "
                    "expected_tool_args (critério positivo), forbidden_literal_ids (Contaminação Tipo A) e forbidden_symbolic_placeholders (Contaminação Tipo B)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "update", "delete"],
                            "description": "Operação sobre o caso de teste.",
                        },
                        "case_id": {
                            "type": "string",
                            "description": "UUID do caso de teste (obrigatório apenas para 'update' ou 'delete').",
                        },
                        "scenario_name": {
                            "type": "string",
                            "description": "Nome descritivo do cenário (ex: 'REF-01: Novo morador Carlos no Condomínio Solar').",
                        },
                        "user_message": {
                            "type": "string",
                            "description": "Mensagem enviada pelo lead/usuário para testar o agente.",
                        },
                        "expected_criteria": {
                            "type": "string",
                            "description": "Rubrica objetiva de aprovação avaliada pelo LLM Juiz (o que a resposta/tool DEVE e NÃO PODE fazer).",
                        },
                        "requires_tool": {
                            "type": "boolean",
                            "description": "Se True, reprova automaticamente caso o agente não acione a ferramenta MCP esperada.",
                        },
                        "expected_tool_name": {
                            "type": "string",
                            "description": "Nome exato da tool MCP esperada (ex: 'getUserByPhone', 'check_availability').",
                        },
                        "is_refutation_case": {
                            "type": "boolean",
                            "description": "True se este caso for um Experimento de Refutação Científica Popperiano (out-of-distribution/adversarial).",
                        },
                        "is_anti_case": {
                            "type": "boolean",
                            "description": "True se este caso for um Anti-Caso de Especificidade S5 (>= 20% da suíte, onde a mutação NÃO deve disparar).",
                        },
                        "forbid_tool_call": {
                            "type": "boolean",
                            "description": "Se True, reprova caso o agente acione qualquer tool MCP (usado em anti-casos de falso positivo).",
                        },
                        "expected_tool_args": {
                            "type": "object",
                            "description": "Mapa de argumentos esperados na chamada da tool (critério positivo S5).",
                        },
                        "forbidden_literal_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Lista de IDs literais de casos antigos que jamais podem aparecer nos argumentos das tools (Contaminação Tipo A).",
                        },
                        "forbidden_symbolic_placeholders": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Lista de placeholders simbólicos (ex: '[singleAccountId]', 'singleAccountId') que não podem ser enviados crus para a tool (Contaminação Tipo B).",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_prompt_eval_suite",
                "description": (
                    "QUANDO USAR: Acione quando o usuário pedir para rodar a bateria de Evals ou quando você quiser validar "
                    "cientificamente o prompt atual contra os casos de teste cadastrados (Quality Gate >= 90%). "
                    "Executa as chamadas MCP reais, o Detector de Contaminação Tipo A/B e o LLM Juiz, retornando o placar completo e atualizando o eval_score da versão."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_live_editor_prompt": {
                            "type": "boolean",
                            "description": "Se True (padrão), avalia o texto em tempo real do editor (incluindo alterações recém-feitas).",
                        },
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
    "update_mindflow_engineering_doc": "✍️ Atualizou Doc de Hipóteses/Memória Compartilhada",
    "list_prompt_eval_datasets": "📋 Listou Casos de Teste (Evals)",
    "manage_prompt_eval_case": "🧪 Gerenciou Caso de Teste / Refutação (Evals)",
    "run_prompt_eval_suite": "🚀 Executou Bateria Científica de Evals",
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
    current_prompt = context.get("current_prompt")
    session_factory = context.get("_session_factory")

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
        res = await read_mindflow_engineering_docs_impl(topic=topic, _session_factory=session_factory)
        summary = f"Tópico consultado: {topic} (v{res.get('version', 1)})"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "update_mindflow_engineering_doc":
        doc_key = arguments.get("doc_key", "prompt_hypotheses_ledger")
        content = arguments.get("content", "")
        change_summary = arguments.get("change_summary", "Atualizado pelo Agente Copilot")
        res = await update_mindflow_engineering_doc_impl(
            doc_key=doc_key,
            content=content,
            updated_by="prompt_ops_copilot",
            change_summary=change_summary,
            _session_factory=session_factory,
        )
        summary = f"{doc_key} atualizado para v{res.get('version', 1)}"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "list_prompt_eval_datasets":
        res = await list_prompt_eval_datasets_impl(
            prompt_id=arguments.get("prompt_id") or prompt_id,
            channel=arguments.get("channel") or channel,
            _session_factory=session_factory,
        )
        summary = f"{res.get('total_cases', 0)} caso(s) de Eval listado(s) ({res.get('refutation_cases_count', 0)} de refutação)"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "manage_prompt_eval_case":
        res = await manage_prompt_eval_case_impl(
            action=arguments.get("action", "create"),
            prompt_id=arguments.get("prompt_id") or prompt_id,
            client_id=arguments.get("client_id") or client_id,
            channel=arguments.get("channel") or channel or "whatsapp",
            case_id=arguments.get("case_id"),
            scenario_name=arguments.get("scenario_name"),
            user_message=arguments.get("user_message"),
            input_data=arguments.get("input_data"),
            expected_criteria=arguments.get("expected_criteria"),
            requires_tool=bool(arguments.get("requires_tool", False)),
            expected_tool_name=arguments.get("expected_tool_name"),
            is_refutation_case=bool(arguments.get("is_refutation_case", False)),
            is_anti_case=bool(arguments.get("is_anti_case", False)),
            forbid_tool_call=bool(arguments.get("forbid_tool_call", False)),
            expected_tool_args=arguments.get("expected_tool_args"),
            forbidden_literal_ids=arguments.get("forbidden_literal_ids"),
            forbidden_symbolic_placeholders=arguments.get("forbidden_symbolic_placeholders"),
            author_email="prompt_ops_copilot",
            _session_factory=session_factory,
        )
        act_label = res.get("action", "operado")
        c_name = (res.get("case") or {}).get("scenario_name") or arguments.get("case_id") or ""
        summary = f"Caso {act_label}: {str(c_name)[:40]}"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    if tool_name == "run_prompt_eval_suite":
        res = await run_prompt_eval_suite_impl(
            version_id=arguments.get("version_id") or version_id,
            prompt_id=arguments.get("prompt_id") or prompt_id,
            client_id=arguments.get("client_id") or client_id,
            channel=arguments.get("channel") or channel or "whatsapp",
            live_prompt_text=arguments.get("live_prompt_text") or current_prompt,
            test_live_editor_prompt=bool(arguments.get("test_live_editor_prompt", True)),
            _session_factory=session_factory,
        )
        pct = res.get("approval_pct", 0)
        passed = res.get("passed_cases", 0)
        total = res.get("total_cases", 0)
        summary = f"Score: {pct}% ({passed}/{total} aprovados)"
        return {"output": res, "badge": {"tool_name": tool_name, "label": label, "summary": summary}}

    return {
        "output": {"error": f"Tool '{tool_name}' desconhecida."},
        "badge": {"tool_name": tool_name, "label": label, "summary": "Tool não encontrada"},
    }


