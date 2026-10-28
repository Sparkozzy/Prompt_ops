import json
import re
import time
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from app.config import settings
from app.services.llm_service import get_openai_client, render_prompt_template, generate_chat_response
from app.services.mcp_service import get_client_mcp_urls, get_client_tools_and_map


def check_contamination_in_tool_calls(
    tool_calls_executed: List[Dict[str, Any]],
    forbidden_literal_ids: Optional[List[str]] = None,
    forbidden_symbolic_placeholders: Optional[List[str]] = None
) -> (bool, Optional[str]):
    """
    Inspeciona os argumentos das ferramentas executadas buscando contaminação:
    - Tipo A: Vazamento de IDs literais de casos antigos (ex: '1277', '275983').
    - Tipo B: Vazamento de placeholders simbólicos não resolvidos (ex: 'singleAccountId', '[singleAccountId]').
    """
    if not tool_calls_executed:
        return False, None

    for t in tool_calls_executed:
        tool_name = t.get("tool_name", "unknown")
        args_str = json.dumps(t.get("input", {}), ensure_ascii=False)

        # 1. Checagem Tipo A (IDs Literais de Casos Antigos)
        if forbidden_literal_ids:
            for bad_id in forbidden_literal_ids:
                if f'"{bad_id}"' in args_str or f"'{bad_id}'" in args_str or bad_id in args_str:
                    return True, f"Contaminação Tipo A detectada na tool '{tool_name}': vazamento do ID literal proibido '{bad_id}'."

        # 2. Checagem Tipo B (Placeholders Simbólicos Não Resolvidos)
        if forbidden_symbolic_placeholders:
            for bad_sym in forbidden_symbolic_placeholders:
                if bad_sym in args_str:
                    return True, f"Contaminação Tipo B detectada na tool '{tool_name}': envio do placeholder simbólico não resolvido '{bad_sym}'."

        # 3. Detector automático universal de ponteiros entre colchetes [var] (Salvaguarda S5)
        bracket_match = re.search(r"\[[A-Za-z_][A-Za-z0-9_]*\]", args_str)
        if bracket_match:
            return True, f"Contaminação Tipo B detectada na tool '{tool_name}': envio de ponteiro simbólico entre colchetes '{bracket_match.group(0)}'."

    return False, None


async def evaluate_single_turn_case(
    prompt_text: str,
    model_name: str,
    temperature: float,
    case_input: Dict[str, Any],
    expected_criteria: str,
    prompt_id: Optional[int] = None,
    client_id: Optional[str] = None,
    requires_tool: bool = False,
    expected_tool_name: Optional[str] = None,
    version_id: Optional[Any] = None,
    forbidden_literal_ids: Optional[List[str]] = None,
    forbidden_symbolic_placeholders: Optional[List[str]] = None,
    forbid_tool_call: bool = False,
    expected_tool_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executa a inferência de um caso de teste conectando aos servidores MCP reais,
    valida a execução de ferramentas (incluindo critérios positivos S5 e anti-casos)
    e submete o resultado ao LLM Juiz.
    """
    start_time = time.time()
    user_msg = case_input.get("user_message", "")
    variables = case_input.get("variables", {})
    history = case_input.get("chat_history", [])

    rendered_system_prompt = render_prompt_template(prompt_text, variables)

    # 1. Carrega as tools reais dos servidores MCP configurados para este cliente/prompt
    mcp_urls = await get_client_mcp_urls(prompt_id=prompt_id, client_id=client_id)
    tools, tool_map = await get_client_tools_and_map(mcp_urls)

    # 2. Executa a resposta do modelo com suporte real a tools
    resp = await generate_chat_response(
        system_prompt=rendered_system_prompt,
        user_message=user_msg,
        chat_history=history,
        model=model_name,
        temperature=temperature,
        tools=tools,
        mcp_tool_map=tool_map,
        version_id=version_id
    )
    generated_text = resp["content"]
    tool_calls_executed = resp.get("tool_calls_executed", [])

    # 3. Validação determinística do acionamento de ferramentas
    tool_called_name = None
    tool_call_valid = None

    if requires_tool:
        executed_names = [t["tool_name"] for t in tool_calls_executed]
        if expected_tool_name and expected_tool_name in executed_names:
            tool_called_name = expected_tool_name
            tool_call_valid = True
        elif executed_names:
            tool_called_name = executed_names[0]
            tool_call_valid = False
        else:
            tool_called_name = None
            tool_call_valid = False
    elif tool_calls_executed:
        tool_called_name = tool_calls_executed[0]["tool_name"]
        tool_call_valid = not forbid_tool_call

    # Validação positiva de argumentos esperados (Salvaguarda S5)
    args_mismatch_reason = None
    if expected_tool_args and tool_calls_executed:
        first_input = tool_calls_executed[0].get("input") or {}
        for k, exp_v in expected_tool_args.items():
            if str(first_input.get(k)) != str(exp_v):
                args_mismatch_reason = (
                    f"Critério Positivo S5 violado no argumento '{k}': "
                    f"esperado='{exp_v}', recebido='{first_input.get(k)}'."
                )
                tool_call_valid = False
                break

    # 4. Formata o sumário de histórico recente e tools para contextualizar o LLM Juiz
    if history:
        history_lines = []
        for h in history:
            role = h.get("role", "desconhecido")
            cnt = h.get("content") or ""
            tc = h.get("tool_calls")
            if tc:
                names = [c.get("function", {}).get("name") or c.get("name") for c in tc]
                history_lines.append(f"- [{role.upper()}]: (Chamou tool {', '.join(filter(None, names))}) {cnt}".strip())
            else:
                history_lines.append(f"- [{role.upper()}]: {cnt}")
        chat_history_summary = "\n".join(history_lines)
    else:
        chat_history_summary = "Nenhum histórico anterior (Início de conversa / Cold Start)."

    if tool_calls_executed:
        tools_summary = "\n".join([
            f"- Ferramenta '{t['tool_name']}' acionada com parâmetros: {json.dumps(t['input'], ensure_ascii=False)} | Retorno MCP: {str(t['output'])[:250]}"
            for t in tool_calls_executed
        ])
    else:
        tools_summary = "Nenhuma ferramenta (tool) foi acionada pelo assistente nesta interação."

    # 5. LLM Juiz avalia a resposta e as ferramentas executadas contra a rubrica
    client = get_openai_client()
    judge_system_prompt = (
        "Você é um LLM Juiz imparcial e rigoroso da MindFlow. "
        "Sua função é avaliar se as ferramentas executadas e a resposta final formulada pelo assistente "
        "atenderam estritamente aos critérios esperados de negócio (rubrica), considerando o histórico recente da conversa. "
        "Se o critério exigir acionamento de uma tool específica e ela não tiver sido executada, você DEVE reprovar. "
        "Se o critério proibir acionamento de tool (anti-caso) ou proibir repetição desnecessária de perguntas já respondidas no histórico, aplique o rigor exigido. "
        "Você deve retornar APENAS um JSON no formato: "
        '{"passed": true/false, "score": 0.0-1.0, "reasoning": "explicação concisa e objetiva"}'
    )

    judge_user_content = (
        f"### Histórico Recente da Conversa (Contexto Prévio):\n{chat_history_summary}\n\n"
        f"### Mensagem Atual do Usuário / Lead:\n{user_msg}\n\n"
        f"### Ações de Ferramentas (MCP) Executadas no Turno Atual:\n{tools_summary}\n\n"
        f"### Resposta Final Gerada pelo Bot no Turno Atual:\n{generated_text}\n\n"
        f"### Critério Esperado (Rubrica de Avaliação):\n{expected_criteria}"
    )

    judge_response = await client.chat.completions.create(
        model=settings.EVAL_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_content}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    judge_eval = json.loads(judge_response.choices[0].message.content or "{}")
    passed = bool(judge_eval.get("passed", False))
    reasoning = judge_eval.get("reasoning", "Sem justificativa.")
    execution_time_ms = int((time.time() - start_time) * 1000)

    # Regra estrita: se era Anti-Caso (forbid_tool_call) e disparou tool, reprova por Falso Positivo
    if forbid_tool_call and tool_calls_executed:
        passed = False
        reasoning = f"REPROVADO EM ANTI-CASO (Falso Positivo / Sobre-disparo): a tool '{tool_called_name}' foi acionada quando forbid_tool_call=True. | {reasoning}"

    # Regra estrita: se uma tool era requerida e não foi acionada com sucesso, reprova
    if requires_tool and not tool_call_valid:
        passed = False
        if not tool_calls_executed:
            reasoning = f"A ferramenta obrigatória '{expected_tool_name}' não foi acionada. {reasoning}"
        elif args_mismatch_reason:
            reasoning = f"{args_mismatch_reason} | {reasoning}"
        else:
            reasoning = f"A ferramenta esperada era '{expected_tool_name}', mas foi acionada '{tool_called_name}'. {reasoning}"

    if args_mismatch_reason and passed:
        passed = False
        reasoning = f"{args_mismatch_reason} | {reasoning}"

    # Regra estrita de Prompt Science: detecção de contaminação Tipo A ou Tipo B
    is_contaminated, contamination_reason = check_contamination_in_tool_calls(
        tool_calls_executed=tool_calls_executed,
        forbidden_literal_ids=forbidden_literal_ids,
        forbidden_symbolic_placeholders=forbidden_symbolic_placeholders
    )
    if is_contaminated:
        passed = False
        reasoning = f"REPROVADO POR CONTAMINAÇÃO: {contamination_reason} | Detalhes do Juiz: {reasoning}"

    return {
        "passed": passed,
        "is_contaminated": is_contaminated,
        "contamination_reason": contamination_reason,
        "generated_response": generated_text,
        "judge_reasoning": reasoning,
        "tool_called": tool_called_name,
        "tool_call_valid": tool_call_valid,
        "execution_time_ms": execution_time_ms
    }
