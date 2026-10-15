import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean, DateTime, ForeignKey, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def get_utc_now():
    return datetime.now(timezone.utc)


class PromptVersion(Base):
    """Armazena versões, branches e histórico de alterações dos prompts."""
    __tablename__ = "prompt_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(BigInteger, nullable=False, index=True)  # ID do prompt na tabela oficial Prompts
    client_id = Column(String(100), nullable=False, index=True)
    client_name = Column(String(255), nullable=True)
    version_label = Column(String(50), nullable=False)  # ex: v1.0.0, v1.1.0-draft
    prompt_text = Column(Text, nullable=False)
    model_name = Column(String(100), default="gpt-4o")
    temperature = Column(Float, default=0.7)
    status = Column(String(50), default="draft", index=True)  # draft, in_testing, approved, published, archived
    commit_message = Column(Text, nullable=True)
    author_email = Column(String(255), nullable=False)
    eval_score = Column(Float, nullable=True)  # Último score obtido (0.0 a 1.0)
    
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=get_utc_now, onupdate=get_utc_now, nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=True)

    eval_runs = relationship("EvalRun", back_populates="version", cascade="all, delete-orphan")


class EvalDataset(Base):
    """Casos de teste dedicados por cliente/prompt para avaliação contínua."""
    __tablename__ = "prompt_eval_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(BigInteger, nullable=False, index=True)
    client_id = Column(String(100), nullable=False, index=True)
    scenario_name = Column(String(255), nullable=False)
    test_type = Column(String(50), default="single_turn")  # single_turn, multi_turn
    input_data = Column(JSONB, nullable=False)  # {"user_message": "...", "variables": {...}} ou {"dialog_turns": [...]}
    expected_criteria = Column(Text, nullable=False)  # Rubrica de aceitação
    requires_tool = Column(Boolean, default=False)
    expected_tool_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)


class EvalRun(Base):
    """Registro mestre de execução de baterias de avaliação."""
    __tablename__ = "prompt_eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id = Column(UUID(as_uuid=True), ForeignKey("prompt_versions.id"), nullable=False, index=True)
    prompt_id = Column(BigInteger, nullable=False, index=True)
    status = Column(String(50), default="queued", index=True)  # queued, running, completed, failed
    total_cases = Column(Integer, default=0)
    passed_cases = Column(Integer, default=0)
    approval_rate = Column(Float, default=0.0)  # 0.0 a 1.0
    triggered_by = Column(String(50), default="manual")  # manual, pre_publish
    started_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    version = relationship("PromptVersion", back_populates="eval_runs")
    cases = relationship("EvalCaseResult", back_populates="run", cascade="all, delete-orphan")


class EvalCaseResult(Base):
    """Resultado detalhado de cada caso de teste executado dentro de uma EvalRun."""
    __tablename__ = "prompt_eval_cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("prompt_eval_runs.id"), nullable=False, index=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("prompt_eval_datasets.id"), nullable=False)
    passed = Column(Boolean, nullable=False)
    generated_response = Column(Text, nullable=False)
    judge_reasoning = Column(Text, nullable=False)
    tool_called = Column(String(100), nullable=True)
    tool_call_valid = Column(Boolean, nullable=True)
    execution_time_ms = Column(Integer, default=0)

    run = relationship("EvalRun", back_populates="cases")


class ToolExecutionLog(Base):
    """Log de auditoria das ferramentas reais (MCP) executadas durante testes."""
    __tablename__ = "tool_execution_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    mcp_server_url = Column(String(500), nullable=False)
    tool_name = Column(String(100), nullable=False, index=True)
    input_payload = Column(JSONB, nullable=False)
    output_payload = Column(JSONB, nullable=True)
    status = Column(String(50), default="success")  # success, error
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)


class MetaGeneratorConfig(Base):
    """Configuração e versionamento do agente Gerador de Prompts."""
    __tablename__ = "meta_generator_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_label = Column(String(50), nullable=False)
    system_prompt = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
