# PromptOps Backend — MindFlow

Serviço de backend em FastAPI para a plataforma **PromptOps** da MindFlow.

Gerencia o ciclo de vida de prompts de IA para múltiplos clientes:
- Versionamento tipo Git (`draft` -> `evals` -> `published`)
- Arena de testes A/B com streaming simultâneo (SSE)
- Execução e inspeção de ferramentas reais (MCP) em tempo real
- Pipeline assíncrono de avaliações (single-turn e multi-turn) com LLM Juiz
- Quality Gate exigindo **>= 90%** de aprovação para promoção a produção
- Sincronização atômica com a tabela oficial `Prompts` no Supabase e rollback em 1 clique

---

## Estrutura do Repositório

```
Prompt_ops/
├── docs/
│   ├── architecture.md    # Arquitetura detalhada, fluxos e modelos de banco
│   ├── workflow.md        # Ciclo de vida e regras de transição de versão
│   └── conventions.md     # Convenções EDW, stack e padrões de código
├── future_ideas.md        # Backlog de ideias futuras para a plataforma
├── app/
│   ├── main.py            # Ponto de entrada da aplicação FastAPI
│   ├── config.py          # Configurações de ambiente (pydantic-settings)
│   ├── database.py        # Conexão com PostgreSQL dedicado + Supabase Singleton
│   ├── models.py          # Modelos SQLAlchemy para o banco dedicado
│   ├── schemas.py         # Schemas de validação Pydantic
│   ├── routers/           # Rotas da API (prompts, arena, evals, generator)
│   └── services/          # Orquestração de LLMs, MCPs, evals e deploy
├── worker.py              # ARQ Worker para processamento assíncrono de evals
├── requirements.txt
├── .env.example
└── README.md
```

---

## Execução Local

```bash
# 1. Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente
cp .env.example .env

# 4. Iniciar a API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Iniciar o Worker de Evals (em outro terminal)
arq worker.WorkerSettings
```
