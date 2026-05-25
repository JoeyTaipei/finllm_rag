import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import reciprocal_rank_fusion


def _doc(content: str, page: int = 0) -> Document:
    return Document(page_content=content, metadata={"source": "test", "page": page})


def test_empty_inputs():
    result = reciprocal_rank_fusion([], [])
    assert result == []


def test_single_doc_unchanged():
    doc = _doc("洗錢防制法第22條", page=1)
    result = reciprocal_rank_fusion([doc], [])
    assert len(result) == 1
    assert result[0].page_content == "洗錢防制法第22條"


def test_rerank_score_in_metadata():
    bm25 = [_doc(f"文件BM25_{i}", page=i) for i in range(5)]
    vec = [_doc(f"文件BM25_{i}", page=i) for i in range(3)]  # overlap with bm25
    result = reciprocal_rank_fusion(bm25, vec)
    assert len(result) > 0
    for doc in result:
        assert "rerank_score" in doc.metadata, "rerank_score missing from metadata"
        assert isinstance(doc.metadata["rerank_score"], float)
        assert doc.metadata["rerank_score"] > 0.0


def test_top_k_respected():
    docs = [_doc(f"doc_{i}", page=i) for i in range(10)]
    result = reciprocal_rank_fusion(docs, docs, top_n=3)
    assert len(result) == 3
