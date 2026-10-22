from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID


# ============================================================
# Schemas: Versões de Prompt
# ============================================================
class PromptVersionBase(BaseModel):
    prompt_id: int
    client_id: str
    client_name: Optional[str] = None
    channel: str = "whatsapp"
    version_label: str
    prompt_text: str
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    commit_message: Optional[str] = None


class PromptVersionCreate(PromptVersionBase):
    author_email: str


class PromptCreateRequest(BaseModel):
    client_name: str
    prompt_name: Optional[str] = "atendimento_geral"
    channel: str = "whatsapp"
    prompt_text: Optional[str] = ""
    author_email: Optional[str] = "admin@mindflow.ai"
    client_id: Optional[str] = None


class PromptCreateResponse(BaseModel):
    id: int
    created_at: Optional[str] = None
    client_name: str = Field(..., alias="Nome do cliente")
    prompt_name: Optional[str] = Field(None, alias="Pormpt_Name")
    channel: str
    prompt_text: str = Field(..., alias="Prompt_Text")
    channels: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class PromptVersionRead(PromptVersionBase):
    id: UUID
    status: str
    author_email: str
    eval_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================
# Schemas: Arena de Testes (Playground A/B)
# ============================================================
class ArenaSingleTestRequest(BaseModel):
    version_id: UUID
    user_message: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class ArenaCompareRequest(BaseModel):
    version_a_id: UUID
    version_b_id: UUID
    user_message: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class ToolCallDetail(BaseModel):
    tool_name: str
    input_payload: Dict[str, Any]
    output_payload: Optional[Dict[str, Any]] = None
    latency_ms: int
    status: str


class ArenaTurnResponse(BaseModel):
    version_id: UUID
    version_label: str
    assistant_message: str
    tool_calls: List[ToolCallDetail] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0


# ============================================================
# Schemas: Datasets e Casos de Avaliação
# ============================================================
class EvalDatasetCreate(BaseModel):
    prompt_id: int
    client_id: str
    channel: str = "whatsapp"
    scenario_name: str
    test_type: str = "single_turn"  # single_turn ou multi_turn
    input_data: Dict[str, Any]
    expected_criteria: str
    requires_tool: bool = False
    expected_tool_name: Optional[str] = None
    created_by: Optional[str] = None
    is_refutation_case: bool = False
    forbidden_literal_ids: List[str] = Field(default_factory=list)
    forbidden_symbolic_placeholders: List[str] = Field(default_factory=list)


class EvalDatasetUpdate(BaseModel):
    scenario_name: Optional[str] = None
    channel: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    expected_criteria: Optional[str] = None
    requires_tool: Optional[bool] = None
    expected_tool_name: Optional[str] = None
    is_active: Optional[bool] = None
    updated_by: Optional[str] = None
    is_refutation_case: Optional[bool] = None
    forbidden_literal_ids: Optional[List[str]] = None
    forbidden_symbolic_placeholders: Optional[List[str]] = None


class EvalDatasetDelete(BaseModel):
    deleted_by: str


class EvalDatasetRead(EvalDatasetCreate):
    id: UUID
    is_active: bool
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================
# Schemas: Execução de Evals e Quality Gate
# ============================================================
class EvalRunTrigger(BaseModel):
    version_id: UUID
    triggered_by: str = "manual"  # manual ou pre_publish


class EvalCaseResultRead(BaseModel):
    dataset_id: UUID
    passed: bool
    generated_response: str
    judge_reasoning: str
    tool_called: Optional[str] = None
    tool_call_valid: Optional[bool] = None
    execution_time_ms: int

    class Config:
        from_attributes = True


class EvalRunDetailRead(BaseModel):
    id: UUID
    version_id: UUID
    prompt_id: int
    status: str
    total_cases: int
    passed_cases: int
    approval_rate: float
    triggered_by: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    cases: List[EvalCaseResultRead] = Field(default_factory=list)

    class Config:
        from_attributes = True


# ============================================================
# Schemas: Publicação em Produção e Rollback
# ============================================================
class PromptPublishRequest(BaseModel):
    version_id: UUID
    author_email: str


class PromptPublishResponse(BaseModel):
    success: bool
    message: str
    prompt_id: int
    version_label: str
    approval_rate: float
    published_at: datetime


class PromptRollbackRequest(BaseModel):
    prompt_id: int
    target_version_id: UUID
    author_email: str


# ============================================================
# Schemas: Agente Gerador de Prompt Corporativo (Autoaprendizado + Context MCP)
# ============================================================
class GeneratorToolConsulted(BaseModel):
    tool_name: str
    label: str
    summary: str
    arguments: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None


class PromptAttachment(BaseModel):
    name: str
    type: str  # 'image' ou 'document'
    mime_type: Optional[str] = None
    data_url: Optional[str] = None  # data:image/png;base64,...
    text_content: Optional[str] = None  # Para documentos de texto/pdf extraídos no client ou server


class MetaGeneratorRequest(BaseModel):
    current_prompt: str
    instruction: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    attachments: Optional[List[PromptAttachment]] = Field(default_factory=list)
    failing_examples: Optional[List[Dict[str, Any]]] = None
    active_memory_xml: Optional[str] = None
    prompt_id: Optional[int] = None
    client_id: Optional[str] = None
    channel: Optional[str] = "whatsapp"
    version_id: Optional[UUID] = None
    arena_history: Optional[Dict[str, List[Dict[str, Any]]]] = None


class MetaGeneratorResponse(BaseModel):
    suggested_prompt: str
    prompt_modified: bool = False
    diff_unified: str  # Diff estilo git (linhas + e -)
    explanation: str
    audit_report: Optional[Dict[str, Any]] = None
    updated_active_memory_xml: Optional[str] = None
    tools_consulted: List[GeneratorToolConsulted] = Field(default_factory=list)

