import pytest

from src.generation.schema import FinLLMAnswer, Source, safe_fallback_answer, validate_answer


def _valid_raw(**overrides) -> dict:
    base = {
        "answer": "測試答案",
        "sources": [],
        "confidence": "high",
        "missing_info": [],
        "reasoning_summary": "測試推理步驟",
        "response_type": "grounded",
    }
    return {**base, **overrides}


def test_valid_answer_passes():
    result = validate_answer(_valid_raw())
    assert result is not None
    assert isinstance(result, FinLLMAnswer)


def test_missing_field_returns_none():
    d = _valid_raw()
    del d["answer"]
    assert validate_answer(d) is None


def test_invalid_confidence_returns_none():
    assert validate_answer(_valid_raw(confidence="invalid")) is None


def test_safe_fallback_is_over_refusal():
    fb = safe_fallback_answer("任何問題")
    assert fb.response_type == "over_refusal"
    assert fb.confidence == "low"


def test_normalized_risk_high():
    ans = validate_answer(_valid_raw(confidence="high"))
    assert ans.normalized_risk == pytest.approx(0.2)


def test_normalized_risk_medium():
    ans = validate_answer(_valid_raw(confidence="medium"))
    assert ans.normalized_risk == pytest.approx(1.0)


def test_normalized_risk_low():
    ans = validate_answer(_valid_raw(confidence="low"))
    assert ans.normalized_risk == pytest.approx(1.8)


def test_supporting_text_truncated():
    src = Source(doc_name="test_doc", page=1, supporting_text="a" * 200)
    assert len(src.supporting_text) == 150
