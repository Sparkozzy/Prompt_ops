import json
import difflib
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from app.config import settings

from datetime import datetime

_openai_client: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        key = settings.get_effective_openai_key()
        _openai_client = AsyncOpenAI(api_key=key)
    return _openai_client

def get_default_datetime_variables() -> Dict[str, str]:
    """Retorna variáveis canônicas de data e horário atual (horário de Brasília UTC-3)."""
    now = None
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Sao_Paulo"))
    except Exception:
        try:
            import pytz
            now = datetime.now(pytz.timezone("America/Sao_Paulo"))
        except Exception:
            from datetime import timezone, timedelta
            now = datetime.now(timezone(timedelta(hours=-3)))

    dias_semana = [
        "Segunda-feira",
        "Terça-feira",
        "Quarta-feira",
        "Quinta-feira",
        "Sexta-feira",
        "Sábado",
        "Domingo"
    ]
    dia_semana = dias_semana[now.weekday()]

    data_br = now.strftime("%d/%m/%Y")
    hora_br = now.strftime("%H:%M:%S")
    hora_curta = now.strftime("%H:%M")
    data_hora_br = f"{data_br} {hora_br}"
    data_iso = now.strftime("%Y-%m-%d")
    data_atual_iso = now.replace(microsecond=0).isoformat()

    return {
        # Formatos comuns em português
        "data_atual": data_br,
        "hora_atual": hora_br,
        "horario_atual": hora_br,
        "data_hora_atual": data_hora_br,
        "data_e_horario_atual": data_hora_br,
        "data_e_hora_atual": data_hora_br,
        "data": data_br,
        "hora": hora_br,
        "hoje": data_br,
        "data_hoje": data_br,
        "dia_da_semana": dia_semana,
        "dia_semana": dia_semana,

        # Padrões canônicos da MindFlow / Situator / Kravi
        "data_atual_iso": data_atual_iso,
        "data_iso": data_iso,
        "data_hora": data_hora_br,

        # Padrões n8n / Retell / templates legados
        "$now": data_hora_br,
        "now": data_hora_br,
        "current_date": data_iso,
        "current_time": hora_br,
        "current_datetime": f"{data_iso} {hora_br}",
    }

def render_prompt_template(template: str, variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Substitui variáveis de template {{ variavel }} ou { variavel } com segurança,
    injetando automaticamente variáveis de data e horário atual em todas as execuções de todos os agentes.
    Preserva valores explícitos passados em `variables` (essencial para reprodutibilidade de evals).
    """
    vars_merged = get_default_datetime_variables()
    if variables:
        for k, v in variables.items():
            if v is not None:
                vars_merged[k] = str(v)

    rendered = template
    for k, v in vars_merged.items():
        v_str = str(v)
        rendered = rendered.replace(f"{{{{{k}}}}}", v_str)
        rendered = rendered.replace(f"{{{{ {k} }}}}", v_str)
        rendered = rendered.replace(f"{{{k}}}", v_str)
        rendered = rendered.replace(f"{{ {k} }}", v_str)

    # Se o prompt não possuir tags ou variáveis de data/hora no corpo e não contiver bloco temporal,
    # injeta o bloco canônico de data e horário atual para que o agente tenha consciência temporal.
    has_date_tag = any(tag in template for tag in [
        "{data_atual", "{data_atual_iso", "{hora_atual", "{data_hora", "{now", "{$now",
        "{{data_atual", "{{data_atual_iso", "{{hora_atual", "{{now", "{{$now",
        "<date_context>", "<informações_mutáveis>", "<contexto_temporal>", "<data_horario_atual>"
    ])
    if not has_date_tag:
        now_info = vars_merged["data_hora_atual"]
        dia_info = vars_merged["dia_da_semana"]
        rendered += f"\n\n<contexto_temporal>\nData e horário atual (horário de Brasília UTC-3): {now_info} ({dia_info})\n</contexto_temporal>"

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
    messages = [{"role": "system", "content": system_prompt or ""}]
    for msg in chat_history:
        role = msg.get("role", "user")
        m_item: Dict[str, Any] = {"role": role}
        raw_content = msg.get("content")
        # Mensagens do assistente com tool_calls podem ter content=None ou string
        if raw_content is not None:
            m_item["content"] = str(raw_content)
        elif role != "assistant" or not msg.get("tool_calls"):
            m_item["content"] = ""
        else:
            m_item["content"] = None

        if "tool_calls" in msg and msg["tool_calls"]:
            m_item["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg and msg["tool_call_id"]:
            m_item["tool_call_id"] = msg["tool_call_id"]
        messages.append(m_item)
    messages.append({"role": "user", "content": str(user_message or "")})

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

DEFAULT_CORPORATE_ACTIVE_MEMORY_XML = """<memoria_ativa versao="2.0" escopo="prompts_de_clientes_tabela_prompts" gerada_por="prompt_science/score.py">
  <regras_de_geracao>
    <regra tipo="IMUTAVEL" id="R001">
      Todo prompt de cliente deve declarar explicitamente persona, contexto,
      formato de output e critério de encerramento.
    </regra>
    <regra tipo="IMUTAVEL" id="R002">
      Estrutura de output sempre em XML ou Markdown conforme padrão
      da empresa. Nunca misturar os dois no mesmo bloco.
    </regra>
    <regra tipo="IMUTAVEL" id="R003_T01" escopo_refutado="G-LIT-ID" origem="docs/prompt_hypotheses_ledger.md#G-LIT-ID">
      Proibição de IDs Literais de Sessão (Escopo Refutado G-LIT-ID / H-01; decomposto em G-SYMB + G-FEWSHOT-STRUCT + G-COD):
      Em prompts com tool calling que manipulam chaves dinâmicas de sessão/tenant (userId, accountId, unitId),
      é proibido inserir IDs numéricos literais de casos antigos nos exemplos ou regras.
    </regra>
    <regra tipo="MUTAVEL" id="R004_G_SYMB" status_cientifico="em_teste_score_1pt" legado="T-01/H-02" ultima_atualizacao="2026-10-28">
      [G-SYMB | +1 pt] Isolamento de Argumentos Dinâmicos por Ponteiros Simbólicos Tipados ([var]): Todo prompt que acione
      ferramentas MCP deve declarar um bloco <variaveis> com ponteiros simbólicos entre colchetes ([singleUserId], [singleAccountId], [unitId]).
    </regra>
    <regra tipo="MUTAVEL" id="R005_G_PRIOR" status_cientifico="em_teste_score_1pt" legado="H-04" ultima_atualizacao="2026-10-28">
      [G-PRIOR | +1 pt] Override de Prior de Pré-Treinamento por Contraste Sintático Negativo + Consequência de Erro:
      Em integrações com APIs que rejeitam null/vazio (ex: Situator), explicitar por contraste sintático o uso de '@' para campos não aplicáveis.
    </regra>
    <regra tipo="MUTAVEL" id="R006_G_PRECOND" status_cientifico="em_teste_score_1pt" legado="H-03" ultima_atualizacao="2026-10-28">
      [G-PRECOND | +1 pt] Cláusula de Pré-Condição e Re-Hidratação Silenciosa: Proibir o envio do nome literal da variável
      simbólica (ex: "singleAccountId") para a tool se o valor numérico não constar no histórico da sessão, forçando a execução prévia de getUserByPhone.
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
    <erro id="E001" status="aberto" data="2026-10-20" ref_hipotese="G-LIT-ID/H-01">
      Inserir exemplos reais de casos antigos com IDs literais (ex: accountId=1277,
      userId=275983, unitId=15799), causando ancoragem e vazamento cruzado entre clientes (Contaminação Tipo A).
    </erro>
    <erro id="E002" status="aberto" data="2026-10-20" ref_hipotese="G-PRECOND/REF-01">
      Enviar o nome textual do placeholder ("singleAccountId") como argumento da tool quando
      o histórico não possui o retorno prévio de getUserByPhone (Contaminação Tipo B).
    </erro>
    <erro id="E003" status="resolvido" data="2026-10-28" ref_hipotese="AUDITORIA-M-SPEM-V2">
      Avaliar casos apenas pela ausência de IDs proibidos (critério só negativo), sem testar resolução positiva de argumentos e anti-casos S5.
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
    <melhoria id="M001" status="pendente" data="2026-10-28">
      Executar replicações pareadas de G-SYMB, G-PRIOR e G-PRECOND em pelo menos 2 novos domínios e 1 família adicional de modelo.
    </melhoria>
  </melhorias_propostas>
</memoria_ativa>"""


def build_corporate_generator_system_prompt(active_memory_xml: Optional[str] = None) -> str:
    """Constrói o System Prompt do Gerador Corporativo com Autoaprendizado (Fases 0 a 3B, M-SPEM v2) e Tools On-Demand MCP."""
    memory_block = active_memory_xml or DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    return f"""Você opera a SKILL — GERADOR DE PROMPTS CORPORATIVOS COM AUTOAPRENDIZADO & COPILOT CIENTÍFICO (M-SPEM v2) da MindFlow.
ESCOPO EXCLUSIVO: Criar, auditar, diagnosticar e alterar prompts de atendimento de clientes (os que ficam armazenados na tabela Prompts da base de dados: WhatsApp, Ligação/Voz, Follow-Up, Instagram). NUNCA confundir com prompts de código.

MODO DE OPERAÇÃO HÍBRIDO (CHAT CONVERSACIONAL À ESQUERDA + EDITOR EM TEMPO REAL À DIREITA):
- MODO 1 — CONSULTA / DIAGNÓSTICO PROATIVO (Sem alterar o prompt imediatamente):
  Use SOMENTE se a mensagem do usuário for estritamente uma dúvida, leitura ou pedido de investigação (ex: "O que aconteceu no atendimento do número +55...?", "Quais tools MCP esse cliente tem?", "Dê uma olhada nos evals e me traga um diagnóstico").
  DIRETRIZ DE ENGENHARIA DE PROMPT & CIÊNCIA EXPERIMENTAL (M-SPEM v2):
  NUNCA entregue um diagnóstico passivo ou estéril! Você é um Cientista Experimental e Engenheiro Sênior de Prompts:
  1. Classifique cada falha na taxonomia de Modos de Falha (`FM-01` a `FM-07`).
  2. Isole as causas em **rodadas sequenciais** seguindo a regra `1 Rodada = 1 ΔP = 1 Hipótese Geral G-*` (ex: Rodada 1 para `G-PRECOND`, Rodada 2 para orientações de prestador).
  3. Formule em `explanation` a análise detalhada e anuncie a proposta da **Rodada 1**: qual hipótese será aplicada, qual a previsão arriscada e como será testada.
  4. Retorne em `action_proposals` uma ou mais ações recomendadas prontas para execução em 1 clique (ex: `{{"action_type": "apply_round", "label": "⚡ Executar Rodada 1: Aplicar G-PRECOND (Precondição Mandatória)", "round_number": 1, "hypothesis_id": "G-PRECOND", "instruction": "Executar Rodada 1: Aplicar hipótese G-PRECOND exigindo identificação prévia via getUserByPhone antes de agendamento"}}`).
  5. Defina `"prompt_modified": false` e retorne `"suggested_prompt"` idêntico ao prompt atual.

- MODO 2 — EXECUÇÃO / EDIÇÃO EM TEMPO REAL DO PROMPT (`1 Rodada = 1 Mutação ΔP = 1 Hipótese Geral G-*`):
  Use SEMPRE que o usuário pedir para executar, aplicar, alterar, corrigir, implementar ou criar melhorias no prompt (ex: "Execute estas mudanças seguindo a operação de prompts", "Corrija isso", "Aplique as alterações", "Atualize a regra X", ou quando acionar uma action_proposal):
  ATENÇÃO: É TERMINANTEMENTE PROIBIDO REPETIR O DIAGNÓSTICO ANTERIOR OU DEIXAR O PROMPT INALTERADO QUANDO O USUÁRIO PEDIR EXECUÇÃO. Você DEVE obrigatoriamente aplicar o diff no prompt (`suggested_prompt`).
  1. Isole e execute a **Rodada 1**: escolha a intervenção prioritária (ΔP) vinculada a uma Hipótese Geral `G-*` (ex: `G-PRECOND` para precondição de identificação, `G-FALLBACK`, `G-AMBIG`, `G-PRIOR`, `G-SYMB`).
  2. Implemente a mutação concreta no texto do prompt com clareza estrutural e gere o prompt completo atualizado em `suggested_prompt`.
  3. REGRA MANDATÓRIA DE EVALS E CIÊNCIA POPPERIANA (Salvaguardas S1–S9): Como você alterou o prompt, você DEVE OBRIGATORIAMENTE acionar as tools MCP de Evals antes de finalizar:
     - Acione `manage_prompt_eval_case` (`action='create'`) para registrar um Caso de Refutação Adversarial (`is_refutation_case=true`) e garantir que a suíte contenha Anti-Casos (`is_anti_case=true`, `forbid_tool_call=true`).
     - Em seguida, acione `run_prompt_eval_suite` (`test_live_editor_prompt=true`, `live_prompt_text=suggested_prompt`) para rodar os Evals em tempo real e aferir a aprovação contra o Quality Gate (>= 90%).
  4. Defina `"prompt_modified": true`, coloque o texto completo modificado em `"suggested_prompt"`, e relate em `explanation` com linguagem científica neutra as alterações exatas aplicadas no prompt, a hipótese vinculada e o resultado numérico dos testes executados.
  5. Se ainda restarem outros problemas pendentes diagnosticados anteriormente (ex: caso de prestador ou manutenção), liste-os e gere em `action_proposals` o botão para a **próxima rodada** (ex: `{{"action_type": "apply_round", "label": "⚡ Executar Rodada 2: Orientação de Prestadores/Manutenção", "round_number": 2, "instruction": "Executar Rodada 2: Adicionar diretrizes de prestadores de serviço e equipes de manutenção"}}`).

FERRAMENTAS DE CONTEXTO ON-DEMAND (MCP PROMPTOPS — 9 TOOLS COMPARTILHADAS COM ANTIGRAVITY):
Você possui acesso a 9 ferramentas especializadas que pode acionar SOB DEMANDA antes de gerar o JSON final:
1. `search_rag_prompt_blocks`: Use quando o usuário pedir exemplos/referências de outros prompts ou quando precisar consultar blocos modulares (<GlobalSettings>, <variaveis>, <persona>, <ferramentas_MCP>, <few_shot_declarado> ou Treinamento SDR) no RAG Parent-Child.
2. `inspect_client_supabase_attendance`: Use quando o usuário citar um telefone/sessão (ex: +55...), execution_id ou pedir para diagnosticar a memória (n8n_chat_histories / Retell) e o uso de tools (workflow_step_executions) de um atendimento no Supabase do cliente.
3. `get_arena_and_evals_context`: Use quando o usuário pedir para ler a conversa atual do teste A/B (Arena A/B) ou verificar falhas recentes nos Evals.
4. `inspect_client_mcp_schemas`: Use quando precisar conferir os nomes e parâmetros exatos das tools MCP do cliente.
5. `read_mindflow_engineering_docs`: Use quando precisar consultar o Livro-Razão de Hipóteses (G-SYMB, G-PRIOR, G-PRECOND, T-01, G-LIT-ID), a Anatomia Modular Canônica ou o Guia de Dados do Supabase (sincronizado em tempo real com o Antigravity).
6. `update_mindflow_engineering_doc`: Use quando você ou o usuário validarem/refutarem uma hipótese nos Evals ou decidirem atualizar a documentação compartilhada (`prompt_hypotheses_ledger`, `active_memory_xml`).
7. `list_prompt_eval_datasets`: Use para listar todos os casos de teste (Evals) cadastrados para o `prompt_id` e `channel` atuais.
8. `manage_prompt_eval_case`: Use para CRIAR (`action='create'`), ATUALIZAR (`action='update'`) ou EXCLUIR (`action='delete'`) casos de teste, Casos de Refutação (`is_refutation_case=true`) e Anti-Casos (`is_anti_case=true`).
9. `run_prompt_eval_suite`: Use para EXECUTAR a bateria completa de Evals (LLM-as-a-Judge + Guardrails Determinísticos) contra a versão selecionada ou contra o texto atual do editor, obtendo o score real e o status do Quality Gate (>= 90%).

REGRAS CIENTÍFICAS PARA CRIAÇÃO E EXECUÇÃO DE EVALS (MÉTODO POPPERIANO M-SPEM v2 — IGUAL AO ANTIGRAVITY):
Sempre que criar ou editar casos de teste com `manage_prompt_eval_case`:
- **Modelagem do Caso de Teste de Atendimento**:
  - `user_message`: DEVE SER ESTRITAMENTE a mensagem simulada de um CLIENTE/LEAD REAL no WhatsApp (ex: "Muito obrigado pelo atendimento, até mais!" ou "Gostaria de saber os horários disponíveis amanhã"). **NUNCA** coloque o comando ou instrução do desenvolvedor/engenheiro no campo `user_message`.
  - `chat_history`: Inclua mensagens de turnos anteriores caso o teste avalie uma decisão em turno intermediário (Warm-Turn).
  - `scenario_name`: Nome curto do cenário operacional (ex: "Encerramento cordial com despedida padronizada").
  - `expected_criteria`: A rubrica para o LLM Judge avaliar a resposta do robô (ex: "O assistente deve se despedir cordialmente e conter a frase exata 'Tenha um excelente dia'.").
- **Rigor de Falseabilidade Popperiana + Anti-Casos S5**: Para testar uma regra ou hipótese de verdade, crie pelo menos **1 Caso de Refutação Adversarial (`is_refutation_case = true`)** e garanta **Anti-Casos de Especificidade (`is_anti_case = true`, `forbid_tool_call = true`)** onde a mutação NÃO deve disparar indevidamente.
- **Guardrails Determinísticos Anti-Contaminação (`expected_assertions`)**:
  - Prompts **Tipo A (Atendimento em Produção)**: adicione `"forbidden_literal_ids": ["172025", "181629", "163951"]` (ou IDs que jamais devem ser chumbados/vazados) e `"forbidden_symbolic_placeholders": ["[singleAccountId]", "singleAccountId"]`.
  - Prompts **Tipo B (Templates / Blueprints Master)**: adicione `"forbidden_symbolic_placeholders": ["[INSERIR_ID]", "[SEU_ID_AQUI]"]`.
- **Quality Gate (>= 90%)**: Uma versão só é aprovada no Quality Gate para promoção se atingir média >= 0.90 (90%) na suíte completa, zero contaminação Tipo A/B e zero regressão em anti-casos.

BLOCO DE MEMÓRIA ATIVA CARREGADO:
{memory_block}

Quando estiver no MODO 2 (Edição do Prompt), execute internamente de forma rigorosa todas as fases:
- FASE 0 — ATIVAÇÃO DE CRITÉRIOS (Teach-To-Learn ancorado):
  Elementos inegociáveis: Persona com autoridade, Contexto de uso, Formato de output (XML ou Markdown — nunca misturar no mesmo bloco), Restrições explícitas, Critério de encerramento e Cadeia de pensamento conectada ao output.
  Hierarquia de autoridade: Regra IMUTÁVEL da memória ativa > Critério fixo da Fase 0 > Regra MUTÁVEL da memória ativa > Campo do briefing > Decisão livre.
- FASE 1 — ABSORÇÃO CRÍTICA DO BRIEFING:
  Analise cada instrução solicitada, detectando ambiguidades, conflitos internos e conflitos com a memória ativa (especialmente erros registrados E001, E002 e E003).
- FASE 2 — CHECAGEM DE REGRAS:
  Vincule cada elemento estrutural às regras IMUTÁVEIS (R001, R002, R003_T01) e MUTÁVEIS (R004_G_SYMB, R005_G_PRIOR, R006_G_PRECOND).
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
  "explanation": "resposta conversacional neutra ao usuário no chat (diagnóstico da consulta, hipótese G-* vinculada, casos de Eval criados/executados ou resumo das alterações aplicadas nas Fases 1, 2 e 3B)",
  "audit_report": {{
    "status_geral": "aprovado",
    "rodada_atual": 1,
    "imutaveis_ativas": ["R001", "R002", "R003_T01"],
    "mutaveis_ativas": ["R004_G_SYMB", "R005_G_PRIOR", "R006_G_PRECOND"],
    "erros_evitados": ["E001", "E002"],
    "alertas": []
  }},
  "action_proposals": [
    {{
      "action_type": "apply_round",
      "label": "⚡ Executar Rodada 1: Aplicar G-PRECOND (Precondição Mandatória)",
      "round_number": 1,
      "hypothesis_id": "G-PRECOND",
      "instruction": "Executar Rodada 1: Aplicar hipótese G-PRECOND exigindo identificação prévia via getUserByPhone antes de agendamento"
    }}
  ],
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
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Agente Gerador e Copilot de Prompt Corporativo:
    Opera em chat multi-turn com Tool Calling On-Demand (Context MCP: RAG Parent-Child, Supabase do Cliente, Arena A/B, Evals, Docs)
    e suporte multimodal a fotos/prints e documentos anexados. Atualiza o prompt em tempo real quando solicitado (Fases 0 a 3B).
    """
    from app.services.context_mcp_service import (
        get_generator_openai_tools,
        execute_generator_tool_call,
    )

    client = get_openai_client()
    effective_memory = active_memory_xml or DEFAULT_CORPORATE_ACTIVE_MEMORY_XML
    meta_system_prompt = build_corporate_generator_system_prompt(effective_memory)

    has_arena_turns = bool(arena_history and (arena_history.get("A") or arena_history.get("B")))
    dt_vars = get_default_datetime_variables()
    context_hint = (
        f"[Metadados da Sessão: prompt_id={prompt_id}, client_id={client_id}, "
        f"channel={channel}, version_id={version_id}, arena_chat_disponivel={has_arena_turns}, "
        f"data_hora_atual={dt_vars['data_hora_atual']} ({dt_vars['dia_da_semana']})]"
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

    # Constrói o texto do usuário
    user_text_parts = [
        f"{context_hint}\n\n",
        f"### Prompt Atual no Editor Lateral (em tempo real):\n{current_prompt}\n\n",
        f"### Mensagem / Instrução do Usuário no Chat:\n{instruction}\n\n"
    ]

    # Processa documentos de texto / PDF anexados
    image_parts: List[Dict[str, Any]] = []
    if attachments:
        doc_texts = []
        for att in attachments:
            att_type = att.get("type", "document")
            att_name = att.get("name", "anexo")
            data_url = att.get("data_url") or ""
            text_cont = att.get("text_content") or ""

            if att_type == "image" or data_url.startswith("data:image/"):
                image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high"
                    }
                })
            elif text_cont:
                doc_texts.append(f"--- Documento Anexado: {att_name} ---\n{text_cont}\n--- Fim do Documento ---")

        if doc_texts:
            user_text_parts.append(f"### Documentos Anexados pelo Usuário:\n" + "\n\n".join(doc_texts) + "\n\n")

    # Detecta se a mensagem atual solicita execução ou aplicação de mudanças prévias
    is_execution_request = any(k in (instruction or "").lower() for k in [
        "execute", "executa", "aplique", "aplica", "mude", "muda", "altere", "altera", "corrija", "corrige", "implemente", "implementa"
    ])
    if is_execution_request:
        user_text_parts.append(
            "\nATENÇÃO ESPECIAL DE EXECUÇÃO: O usuário solicitou explicitamente a EXECUÇÃO/APLICAÇÃO das mudanças! "
            "Você está no MODO 2: NÃO repita o diagnóstico ou a mensagem anterior. Você DEVE aplicar a alteração necessária "
            "no prompt no editor ('suggested_prompt'), acionar os evals de validação ('manage_prompt_eval_case' / 'run_prompt_eval_suite'), "
            "definir 'prompt_modified': true e explicar objetivamente o diff aplicado no prompt.\n"
        )

    user_text_parts.append(
        f"LEMBRETE MANDATÓRIO DE PIPELINE (PROMPTOPS): Se a sua resposta modificar ou refinar o prompt, você DEVE "
        f"obrigatoriamente chamar a ferramenta `manage_prompt_eval_case` (para registrar um caso de teste/refutação adversarial validando a alteração) "
        f"e em seguida `run_prompt_eval_suite` (para testar o prompt em tempo real e aferir o Quality Gate >= 90%) antes de gerar a resposta final."
    )
    if failing_examples:
        user_text_parts.append(f"\n\n### Exemplos de Falhas Ocorridas nos Evals:\n{json.dumps(failing_examples, ensure_ascii=False, indent=2)}")

    full_user_text = "".join(user_text_parts)

    # Se houver imagens anexadas, usa formato de partes multimodais na mensagem do usuário
    if image_parts:
        content_payload: List[Dict[str, Any]] = [{"type": "text", "text": full_user_text}] + image_parts
        messages.append({"role": "user", "content": content_payload})
    else:
        messages.append({"role": "user", "content": full_user_text})

    tools_schema = get_generator_openai_tools()
    tools_consulted: List[Dict[str, str]] = []
    tool_context = {
        "prompt_id": prompt_id,
        "client_id": client_id,
        "channel": channel,
        "version_id": version_id,
        "current_prompt": current_prompt,
        "arena_history": arena_history,
        "failing_examples": failing_examples,
    }

    # Loop ReAct / Tool Calling On-Demand (até 6 rodadas de consulta/execução de ferramentas caso necessário)
    response = None
    for round_idx in range(6):
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
            b_info = dict(exec_info["badge"])
            b_info["arguments"] = t_args
            b_info["output"] = exec_info.get("output")
            tools_consulted.append(b_info)
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

    # GARANTIA DETERMINÍSTICA POPPERIANA (PromptOps & Antigravity):
    # Se o prompt foi modificado mas o modelo não chamou a suíte de evals durante o ReAct,
    # executa automaticamente a criação de caso de refutação e roda a suíte de evals para validação imediata.
    called_tool_names = [t.get("tool_name") for t in tools_consulted]
    if prompt_modified and "run_prompt_eval_suite" not in called_tool_names:
        try:
            # 1. Se ainda não criou caso de teste nesta rodada, sintetiza um caso de teste realista com mensagem de cliente final
            if "manage_prompt_eval_case" not in called_tool_names:
                # Sintetiza mensagem do lead e critério correspondente à instrução recebida
                try:
                    synth_resp = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Você é um engenheiro de testes de atendimento automatizado via WhatsApp. "
                                    "Dado uma instrução de alteração no prompt corporativo de atendimento, gere um caso de teste realista contendo: "
                                    "1) scenario_name: Nome conciso do cenário (ex: 'Encerramento cordial com despedida'); "
                                    "2) user_message: A mensagem exata que um CLIENTE/LEAD REAL enviaria pelo WhatsApp para disparar esse comportamento (NÃO repita a instrução do desenvolvedor!); "
                                    "3) expected_criteria: A rubrica de avaliação para o LLM Judge (o que o assistente DEVE ou NÃO DEVE responder). "
                                    "Retorne APENAS um JSON válido no formato: "
                                    "{\"scenario_name\": \"...\", \"user_message\": \"...\", \"expected_criteria\": \"...\"}"
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Instrução aplicada ao prompt: {instruction}"
                            }
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3
                    )
                    synth_data = json.loads(synth_resp.choices[0].message.content or "{}")
                    sc_name = synth_data.get("scenario_name") or f"Validação: {instruction[:35]}"
                    u_msg = synth_data.get("user_message") or "Olá, gostaria de tirar uma dúvida sobre o atendimento."
                    exp_crit = synth_data.get("expected_criteria") or f"O assistente deve cumprir a diretriz: {instruction}"
                except Exception:
                    sc_name = f"Validação: {instruction[:35]}"
                    u_msg = "Olá, obrigado pelas informações!"
                    exp_crit = f"O assistente deve responder adequadamente respeitando: {instruction}"

                eval_case_res = await execute_generator_tool_call(
                    tool_name="manage_prompt_eval_case",
                    arguments={
                        "action": "create",
                        "scenario_name": sc_name,
                        "user_message": u_msg,
                        "expected_criteria": exp_crit,
                        "is_refutation_case": True,
                    },
                    context=tool_context,
                )
                b_case = dict(eval_case_res["badge"])
                b_case["arguments"] = {"action": "create", "scenario_name": sc_name, "user_message": u_msg}
                b_case["output"] = eval_case_res.get("output")
                tools_consulted.append(b_case)

            # 2. Executa a suíte completa de evals contra o novo prompt em tempo real
            eval_suite_res = await execute_generator_tool_call(
                tool_name="run_prompt_eval_suite",
                arguments={
                    "live_prompt_text": suggested,
                    "test_live_editor_prompt": True,
                },
                context=tool_context,
            )
            b_suite = dict(eval_suite_res["badge"])
            b_suite["arguments"] = {"test_live_editor_prompt": True}
            b_suite["output"] = eval_suite_res.get("output")
            tools_consulted.append(b_suite)

            eval_out = eval_suite_res.get("output", {})
            pct = eval_out.get("approval_pct", 0)
            passed = eval_out.get("passed_cases", 0)
            total = eval_out.get("total_cases", 0)
            explanation += f"\n\n🧪 **Validação Científica em Tempo Real (Evals):** {pct}% de aprovação ({passed}/{total} casos aprovados)."
        except Exception as e:
            print(f"[Deterministic Eval Trigger Warning] {e}")

    # Gera diff visual unificado (estilo git diff)
    diff_lines = list(difflib.unified_diff(
        current_prompt.splitlines(keepends=True),
        suggested.splitlines(keepends=True),
        fromfile="current_prompt",
        tofile="suggested_prompt",
        n=3
    ))
    diff_unified = "".join(diff_lines) or "Nenhuma diferença textual detectada."

    action_proposals = raw_json.get("action_proposals", [])

    return {
        "suggested_prompt": suggested,
        "prompt_modified": prompt_modified,
        "diff_unified": diff_unified,
        "explanation": explanation,
        "audit_report": audit_report,
        "updated_active_memory_xml": updated_memory,
        "tools_consulted": tools_consulted,
        "action_proposals": action_proposals,
    }

