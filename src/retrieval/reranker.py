from langchain_core.documents import Document


def reciprocal_rank_fusion(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    """Merge two ranked doc lists via RRF and return the top-n results."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in (bm25_docs, vector_docs):
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    result_docs = []
    for key in sorted_keys[:top_n]:
        doc = doc_map[key]
        doc.metadata = {**doc.metadata, "rerank_score": scores[key]}
        result_docs.append(doc)
    return result_docs
