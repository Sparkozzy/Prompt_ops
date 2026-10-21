import pytest
from app.services.eval_service import check_contamination_in_tool_calls

def test_check_contamination_type_a():
    """Valida detecção de contaminação Tipo A (IDs literais de casos antigos)."""
    tool_calls = [
        {
            "tool_name": "getSchedulingAreaOptions",
            "input": {"accountId": "1277", "userId": "275983"}
        }
    ]
    forbidden_literals = ["1277", "275983", "15799"]
    forbidden_symbols = ["singleAccountId", "singleUserId"]

    is_contaminated, reason = check_contamination_in_tool_calls(
        tool_calls, forbidden_literals, forbidden_symbols
    )
    assert is_contaminated is True
    assert "Contaminação Tipo A" in reason
    assert "1277" in reason

def test_check_contamination_type_b():
    """Valida detecção de contaminação Tipo B (placeholder simbólico não resolvido)."""
    tool_calls = [
        {
            "tool_name": "getSchedulingAreaOptions",
            "input": {"accountId": "singleAccountId", "userId": "singleUserId"}
        }
    ]
    forbidden_literals = ["1277", "275983", "15799"]
    forbidden_symbols = ["singleAccountId", "singleUserId"]

    is_contaminated, reason = check_contamination_in_tool_calls(
        tool_calls, forbidden_literals, forbidden_symbols
    )
    assert is_contaminated is True
    assert "Contaminação Tipo B" in reason
    assert "singleAccountId" in reason

def test_check_contamination_clean():
    """Valida chamada limpa com novos IDs sem vazamento."""
    tool_calls = [
        {
            "tool_name": "getSchedulingAreaOptions",
            "input": {"accountId": "5501", "userId": "88321"}
        }
    ]
    forbidden_literals = ["1277", "275983", "15799"]
    forbidden_symbols = ["singleAccountId", "singleUserId"]

    is_contaminated, reason = check_contamination_in_tool_calls(
        tool_calls, forbidden_literals, forbidden_symbols
    )
    assert is_contaminated is False
    assert reason is None
