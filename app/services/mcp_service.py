from typing import Dict, Any, List, Optional
import httpx
from app.config import settings
from app.models import ToolExecutionLog, get_utc_now
from app.database import async_session_maker
import time

async def execute_mcp_tool(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    version_id: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executa uma tool em um servidor MCP compatível via HTTP/SSE ou endpoint POST.
    Registra os logs de entrada, saída e latência na tabela tool_execution_logs.
    """
    start_time = time.time()
    status = "success"
    output_payload = None

    try:
        # Se for endpoint FastMCP / SSE JSON-RPC
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "jsonrpc": "2.0",
                "method": f"tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": 1
            }
            # Conecta com Authorization header padrão se necessário
            headers = {}
            if "schedule-service" in mcp_url:
                headers["Authorization"] = "Bearer mf_sk_2026_pre_call_xK9v3Qm7bR4wT1nZ"
                
            response = await client.post(mcp_url, json=payload, headers=headers)
            if response.status_code == 200:
                output_payload = response.json()
            else:
                status = "error"
                output_payload = {"status_code": response.status_code, "body": response.text}
    except Exception as e:
        status = "error"
        output_payload = {"error": str(e)}

    latency_ms = int((time.time() - start_time) * 1000)

    # Persiste log no banco dedicado
    try:
        async with async_session_maker() as session:
            log_entry = ToolExecutionLog(
                version_id=version_id,
                mcp_server_url=mcp_url,
                tool_name=tool_name,
                input_payload=arguments,
                output_payload=output_payload,
                status=status,
                latency_ms=latency_ms,
                created_at=get_utc_now()
            )
            session.add(log_entry)
            await session.commit()
    except Exception as db_err:
        print(f"[ToolLog Warning] Falha ao registrar log de tool: {db_err}")

    return {
        "tool_name": tool_name,
        "input": arguments,
        "output": output_payload,
        "latency_ms": latency_ms,
        "status": status
    }
