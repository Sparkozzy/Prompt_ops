import time
import json
from typing import Dict, Any, List, Optional, Tuple
from mcp.client.sse import sse_client
from mcp import ClientSession
from app.config import settings
from app.models import ToolExecutionLog, get_utc_now
from app.database import async_session_maker, get_production_supabase


def get_mcp_headers(url: str) -> Dict[str, str]:
    """Retorna headers de autenticação requeridos pelo ecossistema MindFlow."""
    return {"Authorization": "Bearer mf_sk_2026_pre_call_xK9v3Qm7bR4wT1nZ"}


async def get_client_mcp_urls(prompt_id: Optional[int] = None, client_id: Optional[str] = None) -> List[str]:
    """
    Descobre dinamicamente os servidores MCP configurados para o cliente
    na tabela client_configurations do Supabase Master.
    """
    try:
        supabase = get_production_supabase()
        if client_id:
            res = supabase.table("client_configurations").select("mcp_urls").eq("client_id", str(client_id)).execute()
            if res.data and res.data[0].get("mcp_urls"):
                return res.data[0]["mcp_urls"]
        if prompt_id:
            res = supabase.table("client_configurations").select("mcp_urls").eq("prompt_id", int(prompt_id)).execute()
            if res.data and res.data[0].get("mcp_urls"):
                return res.data[0]["mcp_urls"]
    except Exception as e:
        print(f"[MCP Discovery Warning] Falha ao consultar mcp_urls do cliente: {e}")

    # Fallback padrão para o schedule service unificado
    return ["https://schedule-service-github.bkpxmb.easypanel.host/sse"]


async def get_client_tools_and_map(mcp_urls: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Conecta aos servidores MCP via SSE, lista as ferramentas disponíveis e as
    formata no padrão OpenAI Function Calling.
    Retorna uma tupla (lista_tools_openai, mapa_tool_nome_para_mcp_url).
    """
    tools = []
    tool_map = {}

    for url in mcp_urls:
        try:
            headers = get_mcp_headers(url)
            async with sse_client(url, headers=headers, timeout=12.0) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    res = await session.list_tools()
                    for t in res.tools:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": t.inputSchema or {"type": "object", "properties": {}}
                            }
                        })
                        tool_map[t.name] = url
        except Exception as e:
            print(f"[MCP Tools Warning] Não foi possível carregar ferramentas de {url}: {e}")

    return tools, tool_map


async def execute_mcp_tool(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    version_id: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executa a tool no servidor MCP real usando o MCP Client SDK via SSE.
    Registra os dados de entrada, saída e latência na tabela tool_execution_logs.
    """
    start_time = time.time()
    status = "success"
    output_payload = None
    headers = get_mcp_headers(mcp_url)

    try:
        async with sse_client(mcp_url, headers=headers, timeout=25.0) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                call_res = await session.call_tool(tool_name, arguments)
                content_list = []
                for item in call_res.content:
                    if hasattr(item, "text"):
                        content_list.append(item.text)
                    else:
                        content_list.append(str(item))
                output_payload = "\n".join(content_list)
    except Exception as e:
        status = "error"
        output_payload = f"Erro na execução da tool '{tool_name}' no MCP: {str(e)}"

    latency_ms = int((time.time() - start_time) * 1000)

    # Persiste log de auditoria no PostgreSQL dedicado do PromptOps
    try:
        async with async_session_maker() as session:
            log_entry = ToolExecutionLog(
                version_id=version_id,
                mcp_server_url=mcp_url,
                tool_name=tool_name,
                input_payload=arguments,
                output_payload={"output": output_payload},
                status=status,
                latency_ms=latency_ms,
                created_at=get_utc_now()
            )
            session.add(log_entry)
            await session.commit()
    except Exception as db_err:
        print(f"[ToolLog Warning] Falha ao persistir log de auditoria da tool: {db_err}")

    return {
        "tool_name": tool_name,
        "input": arguments,
        "output": output_payload,
        "latency_ms": latency_ms,
        "status": status
    }
