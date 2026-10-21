import json
import difflib
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from app.config import settings

_openai_client: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        key = settings.get_effective_openai_key()
        _openai_client = AsyncOpenAI(api_key=key)
    return _openai_client

def render_prompt_template(template: str, variables: Dict[str, Any]) -> str:
    """Substitui variáveis de template {{ variavel }} ou { variavel } com segurança."""
    rendered = template
    for k, v in variables.items():
        rendered = rendered.replace(f"{{{{{k}}}}}", str(v))
        rendered = rendered.replace(f"{{{k}}}", str(v))
    return rendered

async def generate_chat_response(
    system_prompt: str,
    user_message: str,
    chat_history: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    tools: Optional[List[Dict[str, Any]]] = None,
    mcp_tool_map: Optional[Dict[str, str]] = None,
    version_id: Optional[Any] = None
) -> Dict[str, Any]:
    """Gera resposta via OpenAI com suporte a Function Calling real via servidores MCP."""
    client = get_openai_client()
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    if tools:
        kwargs["tools"] = tools

    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    tool_calls_executed = []

    # Se o modelo decidiu acionar ferramentas reais (Function Calling)
    if choice.message.tool_calls:
        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            target_url = mcp_tool_map.get(tool_name) if mcp_tool_map else "https://schedule-service-github.bkpxmb.easypanel.host/sse"

            from app.services.mcp_service import execute_mcp_tool
            exec_res = await execute_mcp_tool(
                mcp_url=target_url,
                tool_name=tool_name,
                arguments=args,
                version_id=version_id
            )
            tool_calls_executed.append(exec_res)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(exec_res.get("output", ""))
            })

        # Segunda chamada ao modelo para formular a resposta final contextualizada com os dados da ferramenta
        second_resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        final_choice = second_resp.choices[0]
        return {
            "content": final_choice.message.content or "",
            "tool_calls_executed": tool_calls_executed,
            "tokens_input": (response.usage.prompt_tokens if response.usage else 0) + (second_resp.usage.prompt_tokens if second_resp.usage else 0),
            "tokens_output": (response.usage.completion_tokens if response.usage else 0) + (second_resp.usage.completion_tokens if second_resp.usage else 0)
        }

    return {
        "content": choice.message.content or "",
        "tool_calls_executed": [],
        "tokens_input": response.usage.prompt_tokens if response.usage else 0,
        "tokens_output": response.usage.completion_tokens if response.usage else 0,
    }

DEFAULT_CORPORATE_ACTIVE_MEMORY_XML = """<memoria_ativa versao="1.0" escopo="prompts_de_clientes_tabela_prompts">
  <regras_de_geracao>
    <regra tipo="IMUTAVEL" id="R001">
      Todo prompt de cliente deve declarar explicitamente persona, contexto,
      formato de output e critério de encerramento.
    </regra>
    <regra tipo="IMUTAVEL" id="R002">
      Estrutura de output sempre em XML ou Markdown conforme padrão
      da empresa. Nunca misturar os dois no mesmo bloco.
    </regra>
    <regra tipo="IMUTAVEL" id="R003_T01">
      Isolamento de Estado por Variáveis Simbólicas: Todo prompt de cliente que acione
      ferramentas MCP deve declarar um bloco <variaveis> com ponteiros simbólicos
      entre colchetes (ex: [singleUserId], [singleAccountId], [unitId]) e jamais conter
      IDs numéricos literais de casos antigos nos exemplos ou regras.
    </regra>
    <regra tipo="MUTAVEL" id="R004_H04" ultima_atualizacao="2026-10-21">
      Em integrações com APIs que rejeitam null/vazio (ex: Situator), explicitar por
      contraste sintático o uso de '@' para campos não aplicáveis em vez de null, None ou "".
    </regra>
    <regra tipo="MUTAVEL" id="R005_H03" ultima_atualizacao="2026-10-21">
      Guardrail de Re-hidratação Silenciosa: Proibir o envio do nome literal da variável
      simbólica (ex: "singleAccountId") para a tool se o valor numérico não constar no
      histórico da sessão, forçando a execução prévia da tool de identificação (getUserByPhone).
    </regra>
  </regras_de_geracao>
  <padroes_de_briefing>
    <padrao id="P001" frequencia="alta">
      Agentes multicanal (WhatsApp em Prompt_Text, Voz/Retell em "Ligação/txt",
      Follow-Up em prompt_fup e Instagram em "Prompt Insta") que exigem mensagens
      curtas, humanizadas e acionamento silencioso de ferramentas MCP via SSE.
    </padrao>
  </padroes_de_briefing>
  <erros_registrados>
    <erro id="E001" status="aberto" data="2026-10-20" ref_hipotese="H-01">
      Inserir exemplos reais de casos antigos com IDs literais (ex: accountId=1277,
      userId=275983, unitId=15799), causando ancoragem e vazamento cruzado entre clientes (Contaminação Tipo A).
    </erro>
    <erro id="E002" status="aberto" data="2026-10-20" ref_hipotese="REF-01">
      Enviar o nome textual do placeholder ("singleAccountId") como argumento da tool quando
      o histórico não possui o retorno prévio de getUserByPhone (Contaminação Tipo B).
    </erro>
  </erros_registrados>
  <preferencias>
    <preferencia tipo="tom" confirmada="sim">
      Direto e técnico nas regras; empático, conciso e natural na fala ao cliente final.
    </preferencia>
    <preferencia tipo="xml" confirmada="sim">
      Tags semânticas (<GlobalSettings>, <variaveis>, <persona>, <ferramentas_MCP>, <few_shot_declarado>).
    </preferencia>
    <preferencia tipo="cadeias_de_pensamento" confirmada="sim">
      Chain of Draft (CoD) interno (<raciocinio_interno_cod>) conectado diretamente à resolução de variáveis e output.
    </preferencia>
  </preferencias>
  <melhorias_propostas>
    <melhoria id="M001" status="pendente" data="2026-10-21">
      Calibrar templates específicos por canal da tabela Prompts (Ligação/Retell vs WhatsApp).
    </melhoria>
  </melhorias_propostas>
</memoria_ativa>"""


def build_corporate_generator_system_prompt(active_memory_xml: Optional[str] = None) -> str:
    """Constrói o System Prompt do Gerador Corporativo com Autoaprendizado (Fases 0 a 3B) e Tools On-Demand MCP."""
    memory_block = active_memory_xml or DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    return f"""Você opera a SKILL — GERADOR DE PROMPTS CORPORATIVOS COM AUTOAPRENDIZADO & COPILOT INTERATIVO da MindFlow.
ESCOPO EXCLUSIVO: Criar, auditar, diagnosticar e alterar prompts de atendimento de clientes (os que ficam armazenados na tabela Prompts da base de dados: WhatsApp, Ligação/Voz, Follow-Up, Instagram). NUNCA confundir com prompts de código.

MODO DE OPERAÇÃO HÍBRIDO (CHAT CONVERSACIONAL À ESQUERDA + EDITOR EM TEMPO REAL À DIREITA):
- MODO 1 — CONSULTA / DIAGNÓSTICO (Sem alterar o prompt):
  Se a mensagem do usuário for apenas uma pergunta ou pedido de investigação (ex: "O que aconteceu no atendimento do número +55...?", "Quais tools MCP esse cliente tem?", "Como os outros prompts no RAG fazem agendamento?", "Por que a versão A foi pior na Arena?"), acione as tools MCP necessárias sob demanda, responda detalhadamente em `explanation`, defina `"prompt_modified": false` e retorne `"suggested_prompt"` idêntico ao prompt atual.
- MODO 2 — EDIÇÃO EM TEMPO REAL DO PROMPT:
  Se o usuário pedir para criar, alterar, corrigir ou aplicar uma melhoria no prompt (ex: "Aplique isso no prompt", "Adicione a regra X", "Corrija com base na conversa da Arena"), execute rigorosamente as Fases 0 a 3B, defina `"prompt_modified": true`, coloque o texto completo atualizado em `"suggested_prompt"` (ele atualizará o editor à direita em tempo real) e explique o que foi feito em `"explanation"`.

FERRAMENTAS DE CONTEXTO ON-DEMAND (MCP PROMPTOPS):
Você possui acesso a 5 ferramentas especializadas que pode acionar SOB DEMANDA antes de gerar o JSON final:
1. `search_rag_prompt_blocks`: Use quando o usuário pedir exemplos/referências de outros prompts ou quando precisar consultar blocos modulares (<GlobalSettings>, <variaveis>, <persona>, <ferramentas_MCP>, <few_shot_declarado> ou Treinamento SDR) no RAG Parent-Child.
2. `inspect_client_supabase_attendance`: Use quando o usuário citar um telefone/sessão (ex: +55...), execution_id ou pedir para diagnosticar a memória (n8n_chat_histories / Retell) e o uso de tools (workflow_step_executions) de um atendimento no Supabase do cliente.
3. `get_arena_and_evals_context`: Use quando o usuário pedir para ler a conversa atual do teste A/B (Arena A/B) ou verificar falhas nos Evals.
4. `inspect_client_mcp_schemas`: Use quando precisar conferir os nomes e parâmetros exatos das tools MCP do cliente.
5. `read_mindflow_engineering_docs`: Use quando precisar consultar o Livro-Razão de Hipóteses (T-01, H-04, H-03, H-01), a Anatomia Modular Canônica ou o Guia de Dados do Supabase.

BLOCO DE MEMÓRIA ATIVA CARREGADO:
{memory_block}

Quando estiver no MODO 2 (Edição do Prompt), execute internamente de forma rigorosa todas as fases:
- FASE 0 — ATIVAÇÃO DE CRITÉRIOS (Teach-To-Learn ancorado):
  Elementos inegociáveis: Persona com autoridade, Contexto de uso, Formato de output (XML ou Markdown — nunca misturar no mesmo bloco), Restrições explícitas, Critério de encerramento e Cadeia de pensamento conectada ao output.
  Hierarquia de autoridade: Regra IMUTÁVEL da memória ativa > Critério fixo da Fase 0 > Regra MUTÁVEL da memória ativa > Campo do briefing > Decisão livre.
- FASE 1 — ABSORÇÃO CRÍTICA DO BRIEFING:
  Analise cada instrução solicitada, detectando ambiguidades, conflitos internos e conflitos com a memória ativa (especialmente erros registrados E001 e E002).
- FASE 2 — CHECAGEM DE REGRAS:
  Vincule cada elemento estrutural às regras IMUTÁVEIS (R001, R002, R003_T01) e MUTÁVEIS (R004_H04, R005_H03).
- FASE 3A — GERAÇÃO DO PROMPT:
  Gere o prompt corporativo completo respeitando a hierarquia e evitando ativamente E001 e E002.
- FASE 3B — AUDITORIA INTERNA (Self-Refine, máx 2 rodadas):
  Verifique se todos os elementos inegociáveis estão presentes, se não há contradições e se o CoD está conectado ao output.
- ATUALIZAÇÃO DA MEMÓRIA ATIVA:
  Atualize o bloco <memoria_ativa> caso uma nova regra mutável, preferência ou erro tenha sido identificado.

Retorne APENAS um objeto JSON válido com a seguinte estrutura exata:
{{
  "prompt_modified": true,
  "suggested_prompt": "texto completo do prompt corporativo gerado (ou o texto original inalterado se prompt_modified=false)",
  "explanation": "resposta conversacional ao usuário no chat (diagnóstico da consulta ou resumo executivo das alterações aplicadas nas Fases 1, 2 e 3B)",
  "audit_report": {{
    "status_geral": "aprovado",
    "rodada_atual": 1,
    "imutaveis_ativas": ["R001", "R002", "R003_T01"],
    "mutaveis_ativas": ["R004_H04", "R005_H03"],
    "erros_evitados": ["E001", "E002"],
    "alertas": []
  }},
  "updated_active_memory_xml": "<memoria_ativa ...>...</memoria_ativa>"
}}"""


async def generate_prompt_meta_suggestion(
    current_prompt: str,
    instruction: str,
    failing_examples: Optional[List[Dict[str, Any]]] = None,
    active_memory_xml: Optional[str] = None,
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    channel: Optional[str] = "whatsapp",
    version_id: Optional[Any] = None,
    arena_history: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Agente Gerador e Copilot de Prompt Corporativo:
    Opera em chat multi-turn com Tool Calling On-Demand (Context MCP: RAG Parent-Child, Supabase do Cliente, Arena A/B, Evals, Docs)
    e atualiza o prompt em tempo real quando solicitado (Fases 0 a 3B).
    """
    from app.services.context_mcp_service import (
        get_generator_openai_tools,
        execute_generator_tool_call,
    )

    client = get_openai_client()
    effective_memory = active_memory_xml or DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    meta_system_prompt = build_corporate_generator_system_prompt(effective_memory)

    has_arena_turns = bool(arena_history and (arena_history.get("A") or arena_history.get("B")))
    context_hint = (
        f"[Metadados da Sessão: prompt_id={prompt_id}, client_id={client_id}, "
        f"channel={channel}, version_id={version_id}, arena_chat_disponivel={has_arena_turns}]"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": meta_system_prompt},
    ]

    if chat_history:
        for msg in chat_history[-12:]:
            r = msg.get("role", "user")
            c = msg.get("content", "")
            if r in ("user", "assistant") and c:
                messages.append({"role": r, "content": c})

    user_content = (
        f"{context_hint}\n\n"
        f"### Prompt Atual no Editor Lateral (em tempo real):\n{current_prompt}\n\n"
        f"### Mensagem / Instrução do Usuário no Chat:\n{instruction}"
    )
    if failing_examples:
        user_content += f"\n\n### Exemplos de Falhas Ocorridas nos Evals:\n{json.dumps(failing_examples, ensure_ascii=False, indent=2)}"

    messages.append({"role": "user", "content": user_content})

    tools_schema = get_generator_openai_tools()
    tools_consulted: List[Dict[str, str]] = []
    tool_context = {
        "prompt_id": prompt_id,
        "client_id": client_id,
        "channel": channel,
        "version_id": version_id,
        "arena_history": arena_history,
        "failing_examples": failing_examples,
    }

    # Loop ReAct / Tool Calling On-Demand (até 3 rodadas de consulta a ferramentas caso necessário)
    response = None
    for round_idx in range(3):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            break

        messages.append(choice.message)
        for tc in tool_calls:
            t_name = tc.function.name
            try:
                t_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                t_args = {}

            exec_info = await execute_generator_tool_call(
                tool_name=t_name,
                arguments=t_args,
                context=tool_context,
            )
            tools_consulted.append(exec_info["badge"])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(exec_info["output"], ensure_ascii=False),
            })
    else:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )

    raw_content = (response.choices[0].message.content if response else None) or "{}"
    try:
        raw_json = json.loads(raw_content)
    except Exception:
        raw_json = {}

    explicit_modified = raw_json.get("prompt_modified")
    suggested = raw_json.get("suggested_prompt")
    if explicit_modified is False or not suggested:
        suggested = current_prompt
        prompt_modified = False
    else:
        prompt_modified = (suggested.strip() != current_prompt.strip())

    explanation = raw_json.get("explanation", "Operação concluída.")
    audit_report = raw_json.get("audit_report", {
        "status_geral": "aprovado" if prompt_modified else "consulta_informativa",
        "rodada_atual": 1,
        "imutaveis_ativas": ["R001", "R002", "R003_T01"],
        "erros_evitados": ["E001", "E002"]
    })
    updated_memory = raw_json.get("updated_active_memory_xml", effective_memory)

    # Gera diff visual unificado (estilo git diff)
    diff_lines = list(difflib.unified_diff(
        current_prompt.splitlines(keepends=True),
        suggested.splitlines(keepends=True),
        fromfile="current_prompt",
        tofile="suggested_prompt",
        n=3
    ))
    diff_unified = "".join(diff_lines) or "Nenhuma diferença textual detectada."

    return {
        "suggested_prompt": suggested,
        "prompt_modified": prompt_modified,
        "diff_unified": diff_unified,
        "explanation": explanation,
        "audit_report": audit_report,
        "updated_active_memory_xml": updated_memory,
        "tools_consulted": tools_consulted,
    }

