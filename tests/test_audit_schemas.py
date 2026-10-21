import pytest
import uuid
from app.models import EvalDataset, PromptVersion
from app.schemas import EvalDatasetCreate, EvalDatasetUpdate

def test_eval_dataset_schema_with_audit_and_science():
    """Valida se o schema EvalDatasetCreate suporta created_by, is_refutation_case e forbidden_literals."""
    payload = EvalDatasetCreate(
        prompt_id=38,
        client_id="kravi_sac",
        channel="whatsapp",
        scenario_name="Teste com IDs Proibidos",
        input_data={"user_message": "Oi"},
        expected_criteria="Deve chamar getUserByPhone",
        created_by="engenheiro@mindflow.com.br",
        is_refutation_case=True,
        forbidden_literal_ids=["1277", "275983"],
        forbidden_symbolic_placeholders=["singleAccountId", "singleUserId"]
    )
    assert payload.created_by == "engenheiro@mindflow.com.br"
    assert payload.is_refutation_case is True
    assert "1277" in payload.forbidden_literal_ids
    assert "singleAccountId" in payload.forbidden_symbolic_placeholders

def test_eval_dataset_update_schema_with_audit():
    """Valida se EvalDatasetUpdate suporta updated_by."""
    payload = EvalDatasetUpdate(
        scenario_name="Atualizado",
        updated_by="auditor@mindflow.com.br"
    )
    assert payload.updated_by == "auditor@mindflow.com.br"
