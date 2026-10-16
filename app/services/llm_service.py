import json
import difflib
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from app.config import settings

_openai_client: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
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

async def generate_prompt_meta_suggestion(
    current_prompt: str,
    instruction: str,
    failing_examples: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, str]:
    """Agente Gerador de Prompt: gera versão alterada e o diff visual unificado."""
    client = get_openai_client()

    meta_system_prompt = (
        "Você é um engenheiro sênior especialista em meta-prompting da MindFlow. "
        "Sua tarefa é aprimorar prompts de agentes conversacionais de voz e texto, "
        "mantendo o tom, regras de negócio e corrigindo falhas apontadas pelo usuário. "
        "Retorne APENAS um JSON no seguinte formato: "
        '{"suggested_prompt": "...", "explanation": "..."}'
    )

    user_content = f"### Prompt Atual:\n{current_prompt}\n\n### Instrução de Alteração:\n{instruction}"
    if failing_examples:
        user_content += f"\n\n### Exemplos de Falhas Ocorridas:\n{json.dumps(failing_examples, ensure_ascii=False, indent=2)}"

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": meta_system_prompt},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )

    raw_json = json.loads(response.choices[0].message.content or "{}")
    suggested = raw_json.get("suggested_prompt", current_prompt)
    explanation = raw_json.get("explanation", "Alterações aplicadas.")

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
        "diff_unified": diff_unified,
        "explanation": explanation
    }
