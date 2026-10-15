import json
import time
from typing import Dict, Any, List
from openai import AsyncOpenAI
from app.config import settings
from app.services.llm_service import get_openai_client, render_prompt_template, generate_chat_response

async def evaluate_single_turn_case(
    prompt_text: str,
    model_name: str,
    temperature: float,
    case_input: Dict[str, Any],
    expected_criteria: str
) -> Dict[str, Any]:
    """
    Executa a inferência de um caso de teste e submete ao LLM Juiz.
    """
    start_time = time.time()
    user_msg = case_input.get("user_message", "")
    variables = case_input.get("variables", {})
    history = case_input.get("chat_history", [])

    rendered_system_prompt = render_prompt_template(prompt_text, variables)

    # 1. Executa a resposta do modelo
    resp = await generate_chat_response(
        system_prompt=rendered_system_prompt,
        user_message=user_msg,
        chat_history=history,
        model=model_name,
        temperature=temperature
    )
    generated_text = resp["content"]

    # 2. LLM Juiz avalia contra a rubrica
    client = get_openai_client()
    judge_system_prompt = (
        "Você é um LLM Juiz imparcial e rigoroso da MindFlow. "
        "Sua função é avaliar se a resposta gerada por um assistente atendeu aos critérios esperados de negócio. "
        "Você deve retornar APENAS um JSON no formato: "
        '{"passed": true/false, "score": 0.0-1.0, "reasoning": "explicação concisa"}'
    )

    judge_user_content = (
        f"### Mensagem do Usuário:\n{user_msg}\n\n"
        f"### Resposta Gerada pelo Bot:\n{generated_text}\n\n"
        f"### Critério Esperado (Rubrica):\n{expected_criteria}"
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

    return {
        "passed": passed,
        "generated_response": generated_text,
        "judge_reasoning": reasoning,
        "execution_time_ms": execution_time_ms
    }
