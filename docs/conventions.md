# Convenções Técnicas — PromptOps Backend

Este documento resume as diretrizes obrigatórias de desenvolvimento para o repositório `Prompt_ops`.

---

## 1. Stack Tecnológica
* **Linguagem:** Python 3.11+
* **Framework Web:** FastAPI (proibido Flask)
* **Processamento Assíncrono / Background:** Redis + ARQ (Async Redis Queue)
* **Chamadas HTTP Externas:** `httpx.AsyncClient` (proibido `requests` síncrono)
* **Conexão com Banco Dedicado:** SQLAlchemy 2.0 (Async) / asyncpg
* **Cliente Supabase:** Instância Singleton global inicializada no `startup` da aplicação
* **Protocolo de Ferramentas:** MCP Python SDK (`mcp` / `FastMCP`) com `AsyncExitStack`

---

## 2. Datas e Horários
* **Persistência em Banco:** Sempre em UTC no formato ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`).
* **Operações de Negócio / Horário de Brasília:** Converter internamente para o fuso `America/Sao_Paulo`.

---

## 3. Schemas e Validação
* Todos os endpoints e retornos de tarefas em background devem utilizar schemas **Pydantic** explícitos localizados em `app/schemas.py`.
* Validação estrita de tipos para payloads de arena, parâmetros de modelos LLM e relatórios de avaliação.
