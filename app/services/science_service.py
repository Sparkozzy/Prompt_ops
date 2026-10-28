import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models import (
    FailureMode,
    ScientificHypothesis,
    ExperimentPreregistration,
    ScientificExperimentRun,
    HypothesisEvent,
    PromptVersionHypothesis,
    PromptVersion,
    GlobalEvalCase,
    PromptEngineeringDoc,
    get_utc_now,
)

# Termos específicos proibidos em enunciados de Hipóteses Gerais (Salvaguarda S9 — Teste do Cliente Apagado)
FORBIDDEN_CLIENT_SPECIFIC_TERMS: List[str] = [
    r"\bkravi\b",
    r"\bhs\s*naturais\b",
    r"\bmy\s*gain\b",
    r"\bsituator\b",
    r"\boslo\b",
    r"\brenato\s*paranhos\b",
    r"\bgetUserByPhone\b",
    r"\bgetUserByCPF\b",
    r"\bgetUserByEmail\b",
    r"\bgetSchedulingAreaOptions\b",
    r"\bgetSchedulingAreaId\b",
    r"\bcreateScheduling\b",
    r"\bloadSituatorAccounts\b",
    r"\bsend_whatsapp_message\b",
    r"\bmake_phone_call\b",
    r"\b3398-8088\b",
    r"\b275983\b",
]

FORBIDDEN_TRIUMPHALIST_PHRASES: List[str] = [
    r"\b100%\s+garantid[oa]\b",
    r"\bcomprovad[oa]\s+definitivamente\b",
    r"\binfal[ií]vel\b",
    r"\bdefinitivamente\s+consolidad[oa]\b",
]

SEED_FAILURE_MODES: List[Dict[str, str]] = [
    {
        "failure_mode_id": "FM-01",
        "name": "Ancoragem Literal e Vazamento Entre Sessões (Cross-Tenant / In-Context Overfitting)",
        "description": "O mecanismo de Self-Attention ancora em valores literais (IDs, nomes, chaves) presentes nos exemplos ou regras do prompt e os reutiliza indevidamente em novas sessões.",
    },
    {
        "failure_mode_id": "FM-02",
        "name": "Conflito com Prior de Pré-Treinamento em Payloads e Formatos",
        "description": "Convenções não-padrão de APIs ou canais (ex: sentinelas para campos vazios, formatos de data/telefone/moeda) competem com o forte viés pré-treinado do modelo (ex: enviar null, '', ISO default).",
    },
    {
        "failure_mode_id": "FM-03",
        "name": "Perda de Estado, Memória Truncada e Placeholders Não Resolvidos",
        "description": "Em conversas multi-turn ou após truncamento de histórico, o modelo perde ponteiros de sessão ou envia o nome textual de variáveis simbólicas ('[var]') como argumento de ferramenta.",
    },
    {
        "failure_mode_id": "FM-04",
        "name": "Descalibração de Gatilho de Ferramenta (Sub-disparo vs. Sobre-disparo)",
        "description": "Desequilíbrio entre precisão e recall no acionamento de tools: omitir chamada quando há sinal implícito (falso negativo) ou acionar prematuramente sem pré-condições (falso positivo).",
    },
    {
        "failure_mode_id": "FM-05",
        "name": "Resolução de Ambiguidade, Múltiplos Candidatos e Cadeia de Fallback",
        "description": "Retorno de múltiplos registros sem chave primária discriminante ou falha no identificador primário exigindo sequência determinística de recuperação ou transbordo.",
    },
    {
        "failure_mode_id": "FM-06",
        "name": "Violação de Restrições de Forma, Comprimento e Prosódia por Canal",
        "description": "Respostas prolixas, múltiplas perguntas no mesmo turno, mistura de XML/Markdown ou formatação visual incompatível com o canal (voz vs. mensageria instantânea).",
    },
    {
        "failure_mode_id": "FM-07",
        "name": "Violação de Escopo, Segurança, LGPD e Invariantes de Negócio",
        "description": "Exposição de dados sensíveis, atuação fora da janela/autoridade operacional ou quebra de guardrails determinísticos.",
    },
]

SEED_HYPOTHESES: List[Dict[str, Any]] = [
    {
        "hypothesis_id": "G-SYMB",
        "legacy_ids": ["T-01", "H-02"],
        "failure_mode_id": "FM-01",
        "title": "Isolamento de Argumentos Dinâmicos por Ponteiros Simbólicos Tipados ([var])",
        "general_statement": "Em prompts que orquestram chamadas de ferramentas com chaves de sessão dinâmicas, substituir todos os valores literais nos exemplos e regras por ponteiros simbólicos entre colchetes ([var]) declarados em um bloco de variáveis elimina o vazamento de dados de exemplos antigos (Contaminação Tipo A) quando o valor real está disponível no contexto da sessão.",
        "scope_conditions": "Prompts com tool calling que recebem IDs dinâmicos por sessão, desde que o retorno da ferramenta de identificação conste no histórico ou haja regra de pré-condição (G-PRECOND).",
        "mechanism": "Remove atratores numéricos/lexicais fixos da janela de atenção do system prompt, forçando o mecanismo de cópia (induction heads) a buscar o valor nos turnos recentes da conversa.",
        "risky_prediction": "Em pareamento A/B com novos registros OOD, o braço Tratamento terá 0% de Contaminação Tipo A (contra >30% no Controle com exemplos literais) mantendo >=95% de acerto nos argumentos quando o estado está presente no histórico.",
        "anti_cases_requirement": "Casos com histórico truncado sem ID resolvido (deve ser combinado com G-PRECOND para evitar Contaminação Tipo B) e turnos puramente conversacionais onde nenhuma ferramenta deve ser acionada.",
    },
    {
        "hypothesis_id": "G-FEWSHOT-STRUCT",
        "legacy_ids": ["T-01", "H-02"],
        "failure_mode_id": "FM-03",
        "title": "Few-Shot de Transição de Estado Explícita (Estado Anterior -> Raciocínio -> Tool -> Estado Atualizado)",
        "general_statement": "Em fluxos conversacionais multi-turn orientados a estado, estruturar exemplos few-shot com blocos explícitos de transição (<estado_sessao_anterior>, <raciocinio_interno_cod>, <chamada_ferramenta>, <estado_sessao_atualizado>) reduz o esquecimento ou troca de etapa da máquina de estados em comparação a exemplos apenas de diálogo superfície.",
        "scope_conditions": "Agentes multi-turn com >= 2 ferramentas encadeadas em sequência lógica.",
        "mechanism": "Explicita a cadeia oculta de transição de estado no in-context learning, reduzindo a ambiguidade sobre qual etapa do protocolo já foi cumprida.",
        "risky_prediction": "Supera o prompt sem exemplos estruturais de transição na seleção da ferramenta correta do turno em >= 15 pontos percentuais em conversas de 3+ turnos.",
        "anti_cases_requirement": "Perguntas fora de ordem (FAQ/dúvida geral no meio do fluxo) onde o modelo deve responder sem perder o estado nem disparar a tool da próxima etapa prematuramente.",
    },
    {
        "hypothesis_id": "G-COD",
        "legacy_ids": ["T-01"],
        "failure_mode_id": "FM-03",
        "title": "Chain of Draft (CoD) Interno Acoplado à Resolução de Parâmetros antes de Tool Call",
        "general_statement": "Exigir um rascunho interno conciso de resolução de variáveis imediatamente conectado à decisão de chamada de ferramenta reduz erros de mapeamento de parâmetros sem degradar o limite de comprimento da resposta ao usuário.",
        "scope_conditions": "Modelos não-reasoning nativos ou chamadas com múltiplos parâmetros condicionais.",
        "mechanism": "Aloca tokens de computação intermediária para vincular variáveis do contexto aos campos do schema antes da emissão do payload JSON.",
        "risky_prediction": "Em ablação controlada (G-SYMB + G-FEWSHOT-STRUCT com vs. sem G-COD), reduz erro de preenchimento de parâmetros complexos sem vazar raciocínio interno no texto visível.",
        "anti_cases_requirement": "Canais de baixa latência (voz síncrona) onde tokens extras antes da fala podem aumentar o TTFT acima do limiar aceitável.",
    },
    {
        "hypothesis_id": "G-PRIOR",
        "legacy_ids": ["H-04"],
        "failure_mode_id": "FM-02",
        "title": "Override de Prior de Pré-Treinamento por Contraste Sintático Negativo + Consequência de Erro",
        "general_statement": "Quando uma API ou contrato de saída exige um marcador ou formato não-padrão que contradiz o hábito de pré-treinamento do modelo (como usar um caractere sentinela em vez de null/vazio, ou formato regional estrito), declarar apenas a instrução positiva falha sob carga; combinar a proibição nominal dos valores default com a consequência técnica do erro e o mapeamento condicional reduz falhas de payload.",
        "scope_conditions": "Integrações de ferramentas com schemas restritivos onde convenções de pré-treinamento (null, None, '', ISO default, código de país com '+') causam rejeição de validação.",
        "mechanism": "Inibe os logits de alta probabilidade a priori ('null', '\"\"') por contraste explícito de atenção negativa pareado ao token substituto.",
        "risky_prediction": "Reduz a taxa de envio de valores default proibidos em campos não aplicáveis para <2% em chamadas de primeira tentativa, superando a instrução apenas positiva.",
        "anti_cases_requirement": "Campos onde o valor real foi informado pelo usuário (garantir que o modelo não sobrescreva dados válidos com o marcador sentinela).",
    },
    {
        "hypothesis_id": "G-PRECOND",
        "legacy_ids": ["H-03"],
        "failure_mode_id": "FM-03",
        "title": "Cláusula de Pré-Condição e Re-Hidratação Silenciosa quando Ponteiro Simbólico é Nulo",
        "general_statement": "Quando um prompt utiliza variáveis simbólicas ([var]) para argumentos obrigatórios de uma ferramenta dependente, adicionar uma invariante explícita que proíbe enviar o nome textual da variável e obriga acionar silenciosamente a ferramenta resolvedora caso o valor real não conste no histórico elimina a Contaminação Tipo B em sessões truncadas.",
        "scope_conditions": "Fluxos onde ferramentas de ação dependem de IDs obtidos por uma ferramenta prévia de busca/identificação e o histórico pode não conter o payload bruto da busca.",
        "mechanism": "Bloqueia o fallback sintático de copiar o identificador da tag <variaveis> para satisfazer o schema JSON obrigatório, redirecionando a decisão para a tool de pré-requisito.",
        "risky_prediction": "Reduz a Contaminação Tipo B (envio de '[var]' ou 'var' literal) de >80% (observado em REF-01/REF-02 sem re-hidratação) para 0% em históricos iniciados no meio da conversa.",
        "anti_cases_requirement": "Sessões onde o ID numérico já consta claramente no histórico recente (o modelo NÃO deve chamar a tool de identificação repetidamente de forma redundante).",
    },
    {
        "hypothesis_id": "G-AMBIG",
        "legacy_ids": ["H-09"],
        "failure_mode_id": "FM-05",
        "title": "Escalonamento Determinístico quando Busca Retorna Múltiplos Candidatos sem Chave de Desambiguação",
        "general_statement": "Quando uma ferramenta de consulta retorna múltiplos cadastros/registros (length > 1) e a resposta da API não expõe chaves primárias relacionais que permitam isolar o registro com segurança nos passos seguintes, proibir a geração de menus de escolha pelo LLM e forçar o transbordo/escalonamento imediato evita erros de vinculação cruzada.",
        "scope_conditions": "Integrações onde o retorno multi-registro omite o identificador único necessário para as tools subsequentes.",
        "mechanism": "Impede que o modelo crie uma promessa conversacional (menu numérico) que o contrato de dados da API não permite cumprir deterministicamente no turno seguinte.",
        "risky_prediction": "Elimina 100% das tentativas de chamada de ferramenta com IDs alucinados ou ausentes em cenários de múltiplos vínculos, sem afetar clientes com vínculo único (length == 1).",
        "anti_cases_requirement": "Casos com registro único (length == 1), nos quais o fluxo automatizado normal deve prosseguir sem transbordo indevido.",
    },
    {
        "hypothesis_id": "G-FALLBACK",
        "legacy_ids": ["H-09"],
        "failure_mode_id": "FM-05",
        "title": "Cadeia Hierárquica Univariada Estrita de Identificação (Um Identificador por Turno)",
        "general_statement": "Em fluxos de localização de cadastro com múltiplas chaves possíveis, impor uma ordem estrita e univariada (tentar chave primária implícita da sessão -> solicitar segunda chave -> solicitar terceira chave -> escalar, pedindo apenas um dado por vez) reduz atrito conversacional e chamadas simultâneas malformadas.",
        "scope_conditions": "Agentes de atendimento que dispõem de >= 2 ferramentas ou chaves de busca de cliente.",
        "mechanism": "Transforma uma decisão aberta de busca em uma árvore de decisão determinística de grau 1 por turno.",
        "risky_prediction": "Aumenta a taxa de localização correta sem pedir dados redundantes no turno inicial e mantém no máximo 1 pergunta por turno.",
        "anti_cases_requirement": "Usuário já fornece espontaneamente a chave secundária na primeira mensagem quando a primária falha (o modelo deve usá-la sem repetir a pergunta).",
    },
    {
        "hypothesis_id": "G-ASYM",
        "legacy_ids": ["H-08"],
        "failure_mode_id": "FM-04",
        "title": "Ampliação de Gatilho de Ferramenta sob Custo Assimétrico de Omissão com Trava de Anti-Casos",
        "general_statement": "Para ferramentas assíncronas de baixo custo de envio e alto custo de omissão (ex: envio de proposta/resumo após apresentação de valor), substituir gatilhos baseados em fechamento completo por gatilhos baseados em qualquer sinal de interesse pós-ancoragem aumenta a captura de leads, desde que limitado por restrições explícitas contra disparo antes da etapa de valor ou diante de recusa.",
        "scope_conditions": "Fluxos comerciais/voz onde a desconexão abrupta perde o lead se a ferramenta assíncrona exigir coleta extensa prévia.",
        "mechanism": "Desloca o limiar de decisão (decision boundary) para maior recall após o marco de apresentação de valor, mantendo bloqueio antes desse marco.",
        "risky_prediction": "Aumenta o recall em casos de interesse parcial/objeção de tempo (meta >= 90%) mantendo a taxa de falso positivo em anti-casos (recusa explícita ou conversa ainda na abertura) <= 5%.",
        "anti_cases_requirement": "Obrigatório >= 30% de anti-casos na bateria: lead recusando a oferta, pedindo para não receber mensagens, ou ainda na fase inicial de qualificação antes dos valores.",
    },
    {
        "hypothesis_id": "G-LIT-ID",
        "legacy_ids": ["H-01"],
        "failure_mode_id": "FM-01",
        "title": "Exemplos Few-Shot com IDs e Chaves Estrangeiras Literais de Casos Antigos",
        "general_statement": "Inserir exemplos concretos contendo IDs numéricos literais de atendimentos reais antigos no prompt de sistema guia melhor o preenchimento de ferramentas em novos atendimentos.",
        "scope_conditions": "ESCOPO REFUTADO: Argumentos dinâmicos de sessão/tenant (userId, accountId, unitId) em chamadas de ferramentas MCP (validado em gpt-4o).",
        "mechanism": "Hipótese original assumia in-context learning benéfico; na prática, o Self-Attention sofre overfitting contextual nos tokens numéricos literais quando o turno atual não exibe um número adjacente óbvio.",
        "risky_prediction": "Refutada: induz vazamento cruzado de IDs entre clientes (Contaminação Tipo A).",
        "anti_cases_requirement": "N/A (Hipótese refutada no escopo de chaves dinâmicas de sessão).",
    },
]

SEED_EVENTS: List[Dict[str, Any]] = [
    {
        "event_id": "evt-0010",
        "event_type": "audit_migration",
        "hypothesis_id": "G-SYMB",
        "notes": "Auditoria Metodológica M-SPEM v2: A antiga T-01 agrupava 3 intervenções distintas (variáveis simbólicas, few-shot estrutural e Chain of Draft) e havia recebido 8 pontos no script legado (scratch/experiment_prompt_science.py) porque o avaliador verificava apenas ausência de IDs proibidos. Nos casos REF-01 e REF-02 (histórico sem retorno de tool), o modelo enviou accountId='singleAccountId' e unitId='unitId' (Contaminação Tipo B), aprovando legitimamente apenas no REF-03 (onde os IDs reais constavam no histórico). Decisão aprovada pelo usuário: recontar honestamente por replicação independente e desmembrar T-01 em G-SYMB, G-FEWSHOT-STRUCT e G-COD.",
        "score_delta": 0,
    },
    {
        "event_id": "evt-0020",
        "event_type": "replication_event",
        "hypothesis_id": "G-LIT-ID",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "n_cases": 2,
        "k_samples": 1,
        "control_pass_rate": 1.0,
        "treatment_pass_rate": 0.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 2,
        "contamination_type_b": 0,
        "mcnemar_p": 0.25,
        "posterior_prob_treatment_better": 0.01,
        "verdict": "replication_fail",
        "score_delta": -4,
        "evidence_ref": "scratch/prompt_science_experiment_results.json#h01",
        "notes": "Evidência histórica migrada (H-01): Vazou IDs literais 1277, 275983 e 15799 nos casos REF-01 (Mariana) e REF-02 (Carlos).",
    },
    {
        "event_id": "evt-0021",
        "event_type": "replication_event",
        "hypothesis_id": "G-SYMB",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "n_cases": 10,
        "k_samples": 1,
        "control_pass_rate": 0.167,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.0156,
        "posterior_prob_treatment_better": 0.996,
        "verdict": "replication_pass",
        "score_delta": 1,
        "evidence_ref": "scratch/eval_variables_result.json + REF-03",
        "notes": "Replicação 1 (Domínio: kravi_sac_whatsapp): Zero Contaminação Tipo A e resolução exata no caso OOD REF-03 (5501, 88321, 4410).",
    },
    {
        "event_id": "evt-0022",
        "event_type": "replication_event",
        "hypothesis_id": "G-FEWSHOT-STRUCT",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "n_cases": 10,
        "k_samples": 1,
        "control_pass_rate": 0.5,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.0625,
        "posterior_prob_treatment_better": 0.985,
        "verdict": "replication_pass",
        "score_delta": 1,
        "evidence_ref": "scratch/eval_few_shot_result.json",
        "notes": "Replicação 1 (Domínio: kravi_sac_whatsapp): Exemplos com transição de estado explícita elevaram aprovação de 50% para 100%.",
    },
    {
        "event_id": "evt-0023",
        "event_type": "replication_event",
        "hypothesis_id": "G-PRIOR",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "n_cases": 3,
        "k_samples": 1,
        "control_pass_rate": 0.0,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.125,
        "posterior_prob_treatment_better": 0.964,
        "verdict": "replication_pass",
        "score_delta": 1,
        "evidence_ref": "Instância Nível 1: H-04 protocolo '@' Situator",
        "notes": "Replicação 1 (Domínio: kravi_sac_whatsapp, Instância H-04): Contraste sintático explícito ('@' vs null/'').",
    },
    {
        "event_id": "evt-0024",
        "event_type": "replication_event",
        "hypothesis_id": "G-PRECOND",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "n_cases": 2,
        "k_samples": 1,
        "control_pass_rate": 0.0,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.25,
        "posterior_prob_treatment_better": 0.95,
        "verdict": "replication_pass",
        "score_delta": 1,
        "evidence_ref": "Instância Nível 1: H-03 getUserByPhone sob demanda",
        "notes": "Replicação 1 (Domínio: kravi_sac_whatsapp, Instância H-03): Eliminou Contaminação Tipo B em históricos truncados.",
    },
    {
        "event_id": "evt-0025",
        "event_type": "replication_event",
        "hypothesis_id": "G-AMBIG",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "n_cases": 2,
        "k_samples": 1,
        "control_pass_rate": 0.0,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.25,
        "posterior_prob_treatment_better": 0.95,
        "verdict": "replication_pass",
        "score_delta": 1,
        "evidence_ref": "Instância Nível 1: H-09 (parte 1 - length > 1 sem PK relacional)",
        "notes": "Replicação 1 (Domínio: kravi_sac_whatsapp): Transbordo determinístico quando a API retorna múltiplos vínculos.",
    },
    {
        "event_id": "evt-0026",
        "event_type": "replication_event",
        "hypothesis_id": "G-FALLBACK",
        "round_mode": "CREATE",
        "domain_id": "kravi_sac_whatsapp",
        "prompt_id": 38,
        "channel": "whatsapp",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "n_cases": 3,
        "k_samples": 1,
        "control_pass_rate": 0.333,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.25,
        "posterior_prob_treatment_better": 0.95,
        "verdict": "replication_pass",
        "score_delta": 1,
        "evidence_ref": "Instância Nível 1: H-09 (parte 2 - hierarquia univariada Telefone -> E-mail -> CPF)",
        "notes": "Replicação 1 (Domínio: kravi_sac_whatsapp): Cadeia univariada estrita sem pedir múltiplos documentos no mesmo turno.",
    },
    {
        "event_id": "evt-0027",
        "event_type": "replication_event",
        "hypothesis_id": "G-ASYM",
        "round_mode": "CREATE",
        "domain_id": "hs_naturais_voz",
        "prompt_id": 10,
        "channel": "ligacao",
        "model_name": "gpt-4o",
        "temperature": 0.2,
        "n_cases": 7,
        "k_samples": 1,
        "control_pass_rate": 0.857,
        "treatment_pass_rate": 1.0,
        "anti_case_regressions": 0,
        "contamination_type_a": 0,
        "contamination_type_b": 0,
        "mcnemar_p": 0.5,
        "posterior_prob_treatment_better": 0.82,
        "verdict": "inconclusive",
        "score_delta": 0,
        "evidence_ref": "scratch/eval_hs_prompt_results.json (Instância Nível 1: H-08)",
        "notes": "Inconclusivo (0 pts) pela Salvaguarda S5: validou recall no TC-07, mas sem >=20% de anti-casos de falso positivo.",
    },
]


def lint_hypothesis_generality(
    hypothesis_id: str,
    title: str,
    general_statement: str,
    scope_conditions: str = "",
) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    if not hypothesis_id.startswith("G-"):
        violations.append(
            f"ID '{hypothesis_id}' inválido para hipótese geral: deve iniciar com 'G-' (ex: G-PRIOR, G-PRECOND)."
        )
    combined_text = f"{title} || {general_statement} || {scope_conditions}"
    for pattern in FORBIDDEN_CLIENT_SPECIFIC_TERMS:
        m = re.search(pattern, combined_text, flags=re.IGNORECASE)
        if m:
            violations.append(
                f"Termo específico proibido no enunciado geral ('{m.group(0)}'). "
                "Mova nomes de clientes/tools/APIs para Instâncias de Nível 1 ou Especificação de Nível 0."
            )
    return (len(violations) == 0), violations


def lint_scientific_language(report_text: str) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    for pattern in FORBIDDEN_TRIUMPHALIST_PHRASES:
        m = re.search(pattern, report_text, flags=re.IGNORECASE)
        if m:
            violations.append(
                f"Expressão triunfalista proibida detectada: '{m.group(0)}'. "
                "Relate apenas tamanho amostral (n×k), efeito (Δ), p-valor/posterior e ameaças à validade."
            )
    return (len(violations) == 0), violations


def compute_prereg_hash(prereg_payload: Dict[str, Any]) -> str:
    canonical_copy = {
        k: v
        for k, v in prereg_payload.items()
        if k not in ("prereg_sha256", "frozen_at", "run_id", "status")
    }
    serialized = json.dumps(canonical_copy, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_and_freeze_prereg(prereg: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    round_mode = str(prereg.get("round_mode", "")).upper()
    if round_mode not in ("CREATE", "APPLY"):
        errors.append("round_mode deve ser 'CREATE' ou 'APPLY'.")

    hypothesis_id = str(prereg.get("hypothesis_id", "")).strip()
    if not hypothesis_id or "," in hypothesis_id or ";" in hypothesis_id or " " in hypothesis_id:
        errors.append("Regra 1 Rodada = 1 Hipótese violada: informe exatamente um hypothesis_id (ex: 'G-PRIOR').")

    if not prereg.get("failure_mode_id"):
        errors.append("failure_mode_id é obrigatório (ex: 'FM-01' a 'FM-07').")
    if not prereg.get("domain_id"):
        errors.append("domain_id (agente/domínio alvo da replicação) é obrigatório.")
    if not prereg.get("delta_p_description"):
        errors.append("delta_p_description (descrição isolada da única mutação ΔP da rodada) é obrigatória.")

    cases = prereg.get("cases") or []
    if len(cases) < 3:
        errors.append(f"A bateria pré-registrada deve conter no mínimo 3 casos (atual: {len(cases)}).")

    anti_cases = [c for c in cases if c.get("is_anti_case") is True]
    min_anti = max(1, math.ceil(0.20 * len(cases))) if cases else 1
    if len(anti_cases) < min_anti:
        errors.append(
            f"Salvaguarda S5 violada: exigido >= 20% de anti-casos (mínimo {min_anti} de {len(cases)}), "
            f"mas foram encontrados {len(anti_cases)}."
        )

    for idx, c in enumerate(cases):
        cid = c.get("id") or f"case_{idx+1}"
        has_positive = bool(
            c.get("expected_tool")
            or c.get("expected_args")
            or c.get("expected_criteria")
            or c.get("forbid_tool_call") is True
        )
        has_negative = bool(
            c.get("forbidden_literal_ids") is not None
            or c.get("forbidden_symbolic_placeholders") is not None
            or c.get("forbid_tool_call") is True
        )
        if not has_positive:
            errors.append(f"Caso '{cid}' sem critério positivo explícito (expected_tool, expected_args ou forbid_tool_call).")
        if not has_negative:
            errors.append(f"Caso '{cid}' sem critério negativo explícito (forbidden_literal_ids/placeholders ou forbid_tool_call).")

    frozen = dict(prereg)
    frozen["prereg_sha256"] = compute_prereg_hash(prereg)
    frozen["frozen_at"] = datetime.now(timezone.utc).isoformat()
    return (len(errors) == 0), errors, frozen


def exact_mcnemar_pvalue(b_treat_only: int, c_ctrl_only: int) -> float:
    n = b_treat_only + c_ctrl_only
    if n == 0:
        return 1.0
    prob = 0.0
    for k in range(b_treat_only, n + 1):
        prob += math.comb(n, k) * (0.5 ** n)
    return round(min(1.0, prob), 6)


def beta_posterior_treatment_better(
    treat_successes: int,
    treat_total: int,
    ctrl_successes: int,
    ctrl_total: int,
    grid_points: int = 400,
) -> float:
    a1, b1 = 1 + max(0, treat_successes), 1 + max(0, treat_total - treat_successes)
    a0, b0 = 1 + max(0, ctrl_successes), 1 + max(0, ctrl_total - ctrl_successes)

    def log_beta_pdf(x: float, a: int, b: int) -> float:
        return (a - 1) * math.log(x) + (b - 1) * math.log(1.0 - x) - (
            math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        )

    step = 1.0 / grid_points
    xs = [(i + 0.5) * step for i in range(grid_points)]
    pdf1 = [math.exp(log_beta_pdf(x, a1, b1)) * step for x in xs]
    pdf0 = [math.exp(log_beta_pdf(x, a0, b0)) * step for x in xs]

    cdf0 = 0.0
    prob_better = 0.0
    for i in range(grid_points):
        prob_better += pdf1[i] * (cdf0 + 0.5 * pdf0[i])
        cdf0 += pdf0[i]

    return round(max(0.0, min(1.0, prob_better)), 4)


def compute_pass_at_k(case_sample_passes: List[Tuple[int, int]], k: int = 3) -> float:
    if not case_sample_passes:
        return 0.0
    vals = []
    for s_i, m_i in case_sample_passes:
        if m_i <= 0:
            continue
        vals.append((s_i / m_i) ** k)
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def evaluate_paired_replication(
    paired_outcomes: List[Dict[str, Any]],
    k_samples: int = 1,
) -> Dict[str, Any]:
    n_cases = len(paired_outcomes)
    if n_cases == 0:
        return {
            "verdict": "inconclusive",
            "score_delta": 0,
            "reason": "Nenhum caso avaliado.",
        }

    b_treat_only = 0
    c_ctrl_only = 0
    anti_regressions = 0
    contam_a = 0
    contam_b = 0
    ctrl_case_passed = 0
    treat_case_passed = 0

    ctrl_samples_tuples: List[Tuple[int, int]] = []
    treat_samples_tuples: List[Tuple[int, int]] = []

    for item in paired_outcomes:
        k = max(1, int(item.get("k_samples", k_samples)))
        c_pass_count = int(item.get("control_passes", 0))
        t_pass_count = int(item.get("treatment_passes", 0))
        ctrl_samples_tuples.append((c_pass_count, k))
        treat_samples_tuples.append((t_pass_count, k))

        c_ok = (c_pass_count == k)
        t_ok = (t_pass_count == k)

        if c_ok:
            ctrl_case_passed += 1
        if t_ok:
            treat_case_passed += 1

        if t_ok and not c_ok:
            b_treat_only += 1
        elif c_ok and not t_ok:
            c_ctrl_only += 1

        if item.get("is_anti_case") and not t_ok:
            anti_regressions += 1

        contam_a += int(item.get("treatment_contamination_a", 0))
        contam_b += int(item.get("treatment_contamination_b", 0))

    ctrl_rate = round(ctrl_case_passed / n_cases, 4)
    treat_rate = round(treat_case_passed / n_cases, 4)
    delta_rate = round(treat_rate - ctrl_rate, 4)

    mcnemar_p = exact_mcnemar_pvalue(b_treat_only, c_ctrl_only)
    posterior_p = beta_posterior_treatment_better(
        treat_case_passed, n_cases, ctrl_case_passed, n_cases
    )
    pass_k_ctrl = compute_pass_at_k(ctrl_samples_tuples, k=k_samples)
    pass_k_treat = compute_pass_at_k(treat_samples_tuples, k=k_samples)

    if contam_a > 0 or contam_b > 0:
        verdict = "replication_fail"
        score_delta = -2
        reason = (
            f"Reprovada por Contaminação no braço Tratamento "
            f"(Tipo A={contam_a}, Tipo B={contam_b}). Penalidade Popperiana: -2 pts."
        )
    elif anti_regressions > 0:
        verdict = "replication_fail"
        score_delta = -2
        reason = (
            f"Reprovada por regressão/falso positivo em {anti_regressions} anti-caso(s) (Salvaguarda S5). "
            "Penalidade Popperiana: -2 pts."
        )
    elif c_ctrl_only > b_treat_only or delta_rate < 0:
        verdict = "replication_fail"
        score_delta = -2
        reason = (
            f"Braço Tratamento inferior ao Controle (Δ={delta_rate:+.1%}, b={b_treat_only}, c={c_ctrl_only}). "
            "Penalidade Popperiana: -2 pts."
        )
    elif (posterior_p >= 0.90 or mcnemar_p <= 0.10) and b_treat_only > c_ctrl_only and treat_rate >= 0.90:
        verdict = "replication_pass"
        score_delta = 1
        reason = (
            f"Replicação aprovada com superioridade pareada "
            f"(Tratamento={treat_rate:.1%} vs Controle={ctrl_rate:.1%}, P(Δ>0)={posterior_p:.3f}, "
            f"McNemar p={mcnemar_p:.4f}, zero regressão em anti-casos). +1 pt."
        )
    else:
        verdict = "inconclusive"
        score_delta = 0
        reason = (
            f"Evidência inconclusiva (poder amostral insuficiente ou empate: "
            f"Tratamento={treat_rate:.1%}, Controle={ctrl_rate:.1%}, P(Δ>0)={posterior_p:.3f}). 0 pts."
        )

    return {
        "n_cases": n_cases,
        "k_samples": k_samples,
        "control_pass_rate": ctrl_rate,
        "treatment_pass_rate": treat_rate,
        "delta_pass_rate": delta_rate,
        "pass_at_k_control": pass_k_ctrl,
        "pass_at_k_treatment": pass_k_treat,
        "b_treatment_only": b_treat_only,
        "c_control_only": c_ctrl_only,
        "anti_case_regressions": anti_regressions,
        "contamination_type_a": contam_a,
        "contamination_type_b": contam_b,
        "mcnemar_p": mcnemar_p,
        "posterior_prob_treatment_better": posterior_p,
        "verdict": verdict,
        "score_delta": score_delta,
        "reason": reason,
    }


async def ensure_scientific_tables_seeded(session: AsyncSession) -> None:
    res_fm = await session.execute(select(FailureMode))
    existing_fm = {r.failure_mode_id for r in res_fm.scalars().all()}
    changed = False
    for fm in SEED_FAILURE_MODES:
        if fm["failure_mode_id"] not in existing_fm:
            session.add(FailureMode(**fm))
            changed = True

    res_h = await session.execute(select(ScientificHypothesis))
    existing_h = {r.hypothesis_id for r in res_h.scalars().all()}
    for h in SEED_HYPOTHESES:
        if h["hypothesis_id"] not in existing_h:
            session.add(ScientificHypothesis(**h))
            changed = True

    res_ev = await session.execute(select(HypothesisEvent))
    existing_ev = {r.event_id for r in res_ev.scalars().all()}
    for ev in SEED_EVENTS:
        if ev["event_id"] not in existing_ev:
            session.add(HypothesisEvent(**ev))
            changed = True

    if changed:
        await session.commit()


async def compute_scientific_state_from_db(session: AsyncSession) -> Dict[str, Any]:
    await ensure_scientific_tables_seeded(session)

    res_fm = await session.execute(select(FailureMode).order_by(FailureMode.failure_mode_id))
    failure_modes = {
        r.failure_mode_id: {
            "failure_mode_id": r.failure_mode_id,
            "name": r.name,
            "description": r.description,
        }
        for r in res_fm.scalars().all()
    }

    res_h = await session.execute(select(ScientificHypothesis))
    hypotheses: Dict[str, Dict[str, Any]] = {}
    for r in res_h.scalars().all():
        hypotheses[r.hypothesis_id] = {
            "hypothesis_id": r.hypothesis_id,
            "legacy_ids": r.legacy_ids or [],
            "failure_mode_id": r.failure_mode_id,
            "title": r.title,
            "general_statement": r.general_statement,
            "scope_conditions": r.scope_conditions,
            "mechanism": r.mechanism or "",
            "risky_prediction": r.risky_prediction or "",
            "anti_cases_requirement": r.anti_cases_requirement or "",
            "score": 0,
            "replications": [],
            "domains_passed": [],
            "models_passed": [],
            "status": "🧪 HIPÓTESE EM TESTE",
        }

    res_ev = await session.execute(select(HypothesisEvent).order_by(HypothesisEvent.created_at))
    audit_migrations: List[Dict[str, Any]] = []
    domains_map: Dict[str, set] = {hid: set() for hid in hypotheses}
    models_map: Dict[str, set] = {hid: set() for hid in hypotheses}

    for ev in res_ev.scalars().all():
        if ev.event_type == "audit_migration":
            audit_migrations.append({
                "event_id": ev.event_id,
                "legacy_id": "T-01",
                "summary": ev.notes or "",
            })
            continue

        hid = ev.hypothesis_id
        if hid in hypotheses:
            h = hypotheses[hid]
            h["score"] += int(ev.score_delta or 0)
            rep_dict = {
                "event_id": ev.event_id,
                "run_id": ev.run_id,
                "round_mode": ev.round_mode,
                "domain_id": ev.domain_id,
                "prompt_id": ev.prompt_id,
                "channel": ev.channel,
                "model_name": ev.model_name,
                "temperature": ev.temperature,
                "n_cases": ev.n_cases,
                "k_samples": ev.k_samples,
                "control_pass_rate": ev.control_pass_rate,
                "treatment_pass_rate": ev.treatment_pass_rate,
                "anti_case_regressions": ev.anti_case_regressions,
                "contamination_type_a": ev.contamination_type_a,
                "contamination_type_b": ev.contamination_type_b,
                "mcnemar_p": ev.mcnemar_p,
                "posterior_prob_treatment_better": ev.posterior_prob_treatment_better,
                "verdict": ev.verdict,
                "score_delta": ev.score_delta,
                "evidence_ref": ev.evidence_ref,
                "notes": ev.notes,
            }
            h["replications"].append(rep_dict)
            if ev.verdict == "replication_pass":
                if ev.domain_id:
                    domains_map[hid].add(str(ev.domain_id))
                if ev.model_name:
                    models_map[hid].add(str(ev.model_name))

    for hid, h in hypotheses.items():
        h["domains_passed"] = sorted(domains_map[hid])
        h["models_passed"] = sorted(models_map[hid])
        score = h["score"]
        if score <= -4:
            h["status"] = "❌ X REFUTADA NO ESCOPO"
        elif score >= 8 and len(h["domains_passed"]) >= 3 and len(h["models_passed"]) >= 2:
            h["status"] = "🏆 TEORIA CONSOLIDADA"
        else:
            h["status"] = "🧪 HIPÓTESE EM TESTE"

    res_runs = await session.execute(
        select(ScientificExperimentRun).where(ScientificExperimentRun.status == "started")
    )
    incomplete_runs = [
        {
            "run_id": r.run_id,
            "hypothesis_id": r.hypothesis_id,
            "domain_id": r.domain_id,
            "started_at": r.started_at.isoformat() if r.started_at else None,
        }
        for r in res_runs.scalars().all()
    ]

    return {
        "failure_modes": failure_modes,
        "hypotheses": hypotheses,
        "audit_migrations": audit_migrations,
        "incomplete_runs": incomplete_runs,
    }
