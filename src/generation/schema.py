from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

_CONF_MAP: dict[str, float] = {"high": 0.9, "medium": 0.5, "low": 0.1}


class Source(BaseModel):
    doc_name: str
    article: Optional[str] = None
    page: int
    supporting_text: str

    @field_validator("supporting_text")
    @classmethod
    def _truncate(cls, v: str) -> str:
        return v[:150]


class FinLLMAnswer(BaseModel):
    answer: str
    sources: list[Source]
    confidence: Literal["high", "medium", "low"]
    missing_info: list[str]
    risk_note: str = (
        "此回答僅供內部法遵或授信分析初步參考，實際決策仍需依最新法規與內部規範確認。"
    )
    reasoning_summary: str
    response_type: Literal["grounded", "hallucination_risk", "over_refusal"]
    normalized_risk: float = 0.0

    @model_validator(mode="after")
    def _compute_normalized_risk(self) -> FinLLMAnswer:
        score = _CONF_MAP[self.confidence]
        self.normalized_risk = 2.0 * (1.0 - score) * 1.0
        return self


def validate_answer(raw: dict) -> FinLLMAnswer | None:
    try:
        return FinLLMAnswer.model_validate(raw)
    except Exception:
        return None


def safe_fallback_answer(query: str) -> FinLLMAnswer:
    return FinLLMAnswer(
        answer=f"無法針對以下問題提供足夠有根據的回答：{query}",
        sources=[],
        confidence="low",
        missing_info=["Retrieved context insufficient or generation failed."],
        reasoning_summary="System fallback due to parse or generation failure.",
        response_type="over_refusal",
    )
