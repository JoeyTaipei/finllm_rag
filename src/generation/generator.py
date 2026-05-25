from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.ingestion.pdf_loader import CHROMA_DIR, COLLECTION, EMBED_MODEL
from src.retrieval.pipeline import HybridRetriever
from src.generation.schema import FinLLMAnswer, safe_fallback_answer, validate_answer

SYSTEM_PROMPT = """You are a Taiwan financial law compliance assistant for KGI Bank.

Rules you MUST follow at all times:
1. You are a Taiwan financial law compliance assistant for KGI Bank.
2. Answer only in Traditional Chinese.
3. NEVER invent law articles, article numbers, or FSC rulings.
4. If an article number is not present in the retrieved context, do not mention it.
5. If retrieved context is insufficient, classify as over_refusal and explain what is missing.
6. Always show your reasoning step by step before giving the final answer.
7. Classify your response as one of: grounded, hallucination_risk, over_refusal.

You MUST respond with ONLY a valid JSON object — no markdown, no prose, no code fences.

JSON schema (all fields required):
{{
  "answer": "<Traditional Chinese answer>",
  "sources": [
    {{
      "doc_name": "<PDF stem name>",
      "article": "<article number if present in context, else null>",
      "page": <page number integer>,
      "supporting_text": "<verbatim excerpt from context, max 150 chars>"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "missing_info": ["<what is missing, if any>"],
  "reasoning_summary": "<step-by-step reasoning in Traditional Chinese>",
  "response_type": "grounded" | "hallucination_risk" | "over_refusal"
}}
"""

_CODE_BLOCK_RE = re.compile(r"`{3}(?:json)?\s*([\s\S]*?)`{3}", re.IGNORECASE)
_RAW_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> str | None:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{"):
            return candidate
    m2 = _RAW_JSON_RE.search(text)
    return m2.group().strip() if m2 else None


def _parse_to_answer(json_str: str | None, query: str) -> FinLLMAnswer:
    if json_str is None:
        return safe_fallback_answer(query)
    try:
        raw_dict = json.loads(json_str)
    except json.JSONDecodeError:
        return safe_fallback_answer(query)
    return validate_answer(raw_dict) or safe_fallback_answer(query)


def _build_retriever() -> HybridRetriever:
    from src.ingestion.pdf_loader import load_and_index

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    documents: list[Document] = []
    store = None

    if CHROMA_DIR.exists():
        store = Chroma(
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding,
        )
        raw = store.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]

    if not documents:
        # Cold start: .chroma/ absent or empty (e.g. Streamlit Cloud).
        # Build index from the committed PDFs in data/raw_pdfs/.
        documents, store = load_and_index(chroma_dir=CHROMA_DIR)

    return HybridRetriever(store, documents, top_k=20)


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = (
            f"[{i}] 來源: {meta.get('source', '未知')} "
            f"| 頁面: {meta.get('page', '?')} "
            f"| 條文: {meta.get('article_number', '未標示')}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# Module-level singletons — initialized once at import time
retriever: HybridRetriever = _build_retriever()

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "問題：{query}\n\n參考文件：\n{context}"),
    ]
)
_chain = _prompt | _llm | StrOutputParser()


@traceable(
    name="finllm_generate",
    run_type="chain",
    metadata={
        "model": "claude-sonnet-4-6",
        "retrieval_strategy": "hybrid_rrf",
    },
    project_name=os.getenv("LANGCHAIN_PROJECT", "finllm-kgi-demo"),
)
def generate(query: str) -> FinLLMAnswer:
    docs = retriever.retrieve(query, top_n=5)
    context = format_context(docs)
    raw_output: str = _chain.invoke({"query": query, "context": context})
    json_str = _extract_json(raw_output)
    answer = _parse_to_answer(json_str, query)

    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.add_metadata(
            {
                "chunk_count": len(docs),
                "response_type": answer.response_type,
                "normalized_risk": answer.normalized_risk,
            }
        )
    return answer
