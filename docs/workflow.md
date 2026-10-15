# Workflow de Engenharia de Prompts (PromptOps Lifecycle)

Este documento estabelece o fluxo de trabalho obrigatório para criação, alteração e publicação de prompts na MindFlow.

---

## Ciclo de Vida do Prompt

```
[Importar de Produção ou Novo] 
         │
         ▼
[Criação de Branch / Draft Local]
         │
         ▼
[Refinamento (Manual ou via Agente Gerador)]
         │
         ▼
[Arena de Teste A/B com MCPs Reais]
         │
         ▼
[Bateria de Evaluations (Single & Multi-turn)]
         │
   ┌─────┴─────┐
   ▼           ▼
Score < 90%   Score >= 90%
(Bloqueado)   (Quality Gate Aprovado)
   │           │
   │           ▼
   │     [Deploy para Produção no Supabase]
   │           │
   │           ▼
   │     [Backup & Tag da Versão Publicada]
   │
   └──► [Voltar ao Refinamento]
```

---

## 1. Importação e Inicialização
1. O engenheiro seleciona o cliente no Hub.
2. A plataforma consulta a tabela `Prompts` no Supabase para buscar o `Prompt_Text` atualmente em produção para aquele cliente.
3. Um registro base é gravado no banco dedicado do PromptOps como `v1.0.0-published` (se for o primeiro acesso).

---

## 2. Criação de Nova Versão (Draft)
1. O engenheiro cria uma nova versão de trabalho (ex: `v1.1.0-draft`).
2. O prompt de produção continua 100% isolado e inalterado.
3. Todas as alterações ocorrem no banco dedicado.

---

## 3. Testes Interativos na Arena
1. O engenheiro testa a versão em modo individual ou em modo comparativo (Arena A/B contra a versão de produção).
2. Se o prompt disparar ferramentas (como agendamento via MCP), a chamada real é executada contra os servidores MCP do cliente.
3. O painel exibe em tempo real o payload enviado à ferramenta e a resposta recebida.
4. O engenheiro pode utilizar o **Agente Gerador de Prompt** para sugerir melhorias baseadas nas respostas obtidas.

---

## 4. Execução de Evaluations & Quality Gate
1. O engenheiro dispara a bateria de testes do cliente (disparada em background via ARQ Worker).
2. A bateria testa todos os cenários cadastrados no dataset (perguntas únicas e diálogos multi-turno com o simulador de leads).
3. O LLM Juiz gera notas individuais e o score consolidado:
   * **Score >= 90%:** O botão **"Publicar em Produção"** é habilitado.
   * **Score < 90%:** O deploy é **bloqueado**. A interface apresenta a lista dos casos que falharam e a justificativa técnica de cada reprovação.

---

## 5. Publicação em Produção (Deploy Atômico)
1. Ao confirmar o deploy:
   * A linha existente do cliente na tabela oficial `Prompts` do Supabase é atualizada (`UPDATE Prompts SET Prompt_Text = ... WHERE id = ...`), mantendo o `id` inalterado.
   * A versão no banco do PromptOps é marcada como `published`.
   * Um snapshot da versão anterior é mantido como backup imediato.
2. Se qualquer anomalia for observada pós-deploy, o engenheiro pode acionar o **Rollback em 1 Clique**, que restaura a versão anterior na tabela `Prompts` em menos de 1 segundo.
