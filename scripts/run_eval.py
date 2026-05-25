from __future__ import annotations

import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example

from src.generation.generator import format_context, generate, retriever

JUDGE_PROMPT = """You are a Taiwan financial AI evaluation expert applying WiDS governance standards.

Question: {input}
AI Answer: {prediction}
Reference Answer: {reference}
Retrieved Context: {context}

Score each dimension 0-10:

1. correctness: Is the answer factually correct per Taiwan AML and Banking law?

2. citation_groundedness: Does every cited article ACTUALLY APPEAR in the retrieved context text?
   Score 0 immediately if the answer mentions any article not present in context.
   This is the most critical check — hallucinated articles in compliance context cause regulatory risk.

3. localization: Does the answer use Taiwan financial terminology correctly?
   Penalize if it references foreign law (SEC, MAS, FCA, US law) without being explicitly asked.

4. hallucination_risk: 10 = zero hallucination, 0 = invented facts or articles.
   Check: does the answer contain specific numbers, dates, or article names not in context?

5. usefulness: Would a KGI compliance officer or credit analyst trust and act on this answer?

Also classify the response:
- grounded: answer fully supported by context
- hallucination_risk: answer contains facts beyond context
- over_refusal: answer refuses a legitimate answerable question

Output ONLY valid JSON, no other text:
{{
  "correctness": 0,
  "citation_groundedness": 0,
  "localization": 0,
  "hallucination_risk": 0,
  "usefulness": 0,
  "overall": 0,
  "response_type": "grounded",
  "critical_error": false,
  "feedback": "one sentence in Traditional Chinese"
}}"""

TEST_CASES = [
    {
        "inputs": {"query": "銀行在什麼情況下可以暫停客戶帳戶功能？"},
        "outputs": {"reference": "洗錢防制法第二十二條"},
    },
    {
        "inputs": {"query": "金融機構辦理KYC需蒐集哪些客戶資料？"},
        "outputs": {"reference": "洗錢防制法主法"},
    },
    {
        "inputs": {"query": "可疑交易申報的時限規定為何？"},
        "outputs": {"reference": "第十七條辦法"},
    },
    {
        "inputs": {"query": "銀行法對於同一人持股比例有何限制？"},
        "outputs": {"reference": "銀行法"},
    },
    {
        "inputs": {"query": "金管會是否規定所有信用卡利率不超過3%？"},
        "outputs": {"reference": "此為負面測試題，正確回答應為 over_refusal，說明該資訊不在文件中。"},
    },
]

DATASET_NAME = "finllm-kgi-eval-v1"

_judge_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
_judge_cache: dict[str, dict] = {}

_RAW_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _run_judge(run: Run, example: Example) -> dict:
    cache_key = f"{run.id}:{example.id}"
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    query = (example.inputs or {}).get("query", "")
    reference = (example.outputs or {}).get("reference", "")
    outputs = run.outputs or {}
    answer_text = outputs.get("answer", "")
    context = outputs.get("context", "")

    filled = JUDGE_PROMPT.format(
        input=query,
        prediction=answer_text,
        reference=reference,
        context=context,
    )
    result = _judge_llm.invoke(filled)
    raw = result.content if hasattr(result, "content") else str(result)

    m = _RAW_JSON_RE.search(raw)
    try:
        parsed = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        parsed = {}

    _judge_cache[cache_key] = parsed
    return parsed


def _score(run: Run, example: Example, key: str) -> dict:
    result = _run_judge(run, example)
    raw_score = result.get(key, 0)
    score = float(raw_score) / 10.0
    feedback = result.get("feedback", "")
    if result.get("critical_error"):
        feedback = f"[CRITICAL ERROR] {feedback}"
    return {"key": key, "score": score, "comment": feedback}


def eval_correctness(run: Run, example: Example) -> dict:
    return _score(run, example, "correctness")


def eval_citation_groundedness(run: Run, example: Example) -> dict:
    return _score(run, example, "citation_groundedness")


def eval_localization(run: Run, example: Example) -> dict:
    return _score(run, example, "localization")


def eval_hallucination_risk(run: Run, example: Example) -> dict:
    return _score(run, example, "hallucination_risk")


def eval_usefulness(run: Run, example: Example) -> dict:
    return _score(run, example, "usefulness")


def _get_or_create_dataset(client: Client) -> str:
    if not client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.create_dataset(DATASET_NAME, description="KGI Bank AML/Banking law eval v1")
        client.create_examples(
            inputs=[tc["inputs"] for tc in TEST_CASES],
            outputs=[tc["outputs"] for tc in TEST_CASES],
            dataset_id=ds.id,
        )
    return DATASET_NAME


def _target(inputs: dict) -> dict:
    query = inputs["query"]
    answer = generate(query)
    ctx = format_context(retriever.retrieve(query, top_n=5))
    return {**answer.model_dump(), "context": ctx}


def main() -> None:
    client = Client()
    dataset_name = _get_or_create_dataset(client)

    results = evaluate(
        _target,
        data=dataset_name,
        evaluators=[
            eval_correctness,
            eval_citation_groundedness,
            eval_localization,
            eval_hallucination_risk,
            eval_usefulness,
        ],
        experiment_prefix="finllm-kgi-eval",
        metadata={
            "dataset_version": "v1",
            "retriever_version": "hybrid_rrf_v1",
            "reranker_type": "reciprocal_rank_fusion",
            "model": "claude-sonnet-4-6",
        },
    )

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / "eval_results.csv"
    df = results.to_pandas()
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Results written to {out_path} ({len(df)} rows)")

    score_cols = [c for c in df.columns if c.startswith("feedback.")]
    if score_cols:
        print(df[["inputs.query", *score_cols]].to_string())


if __name__ == "__main__":
    main()
