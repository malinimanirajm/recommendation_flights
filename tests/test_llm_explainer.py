# tests/test_llm_explainer.py
from unittest.mock import patch
from llm_explainer import explain_recommendation

@patch("llm_explainer.call_llm_api")
def test_explain_recommendation(mock_call):
    mock_call.return_value = "This is a safe explanation."
    result = explain_recommendation("Flight A is faster")
    assert "safe explanation" in result
