from __future__ import annotations

import json
from pathlib import Path

import pytest

BENCHMARK_PATH = Path("data/eval/finllm_kgi_benchmark_v1.jsonl")

_HALL_CATEGORIES = ("non_existent_article", "foreign_law", "false_premise")


def load_negative_cases() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        return []
    cases = []
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if obj["category"] == "negative":
                    cases.append(obj)
    return cases


_NEG_CASES = load_negative_cases()
_HALL_CASES = [c for c in _NEG_CASES if c["negative_category"] in _HALL_CATEGORIES]


@pytest.mark.integration
@pytest.mark.parametrize("case", _NEG_CASES, ids=[c["id"] for c in _NEG_CASES])
def test_negative_not_high_confidence(case):
    from src.generation.generator import generate

    answer = generate(case["query"])
    assert answer.confidence != "high" or len(answer.missing_info) > 0, (
        f"[{case['id']}] Negative case returned high confidence with no missing_info: "
        f"{case['query'][:60]}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("case", _HALL_CASES, ids=[c["id"] for c in _HALL_CASES])
def test_hallucination_cases_not_grounded(case):
    from src.generation.generator import generate

    answer = generate(case["query"])
    assert answer.response_type != "grounded", (
        f"[{case['id']}] Hallucination-risk case classified as grounded: "
        f"{case['query'][:60]}"
    )
