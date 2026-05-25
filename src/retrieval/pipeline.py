import jieba
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.retrieval.reranker import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        chroma_store: Chroma,
        documents: list[Document],
        top_k: int = 20,
    ) -> None:
        self._chroma = chroma_store
        self._top_k = top_k
        tokenized = [list(jieba.cut(doc.page_content)) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._docs = documents

    def retrieve(self, query: str, top_n: int = 5) -> list[Document]:
        query_tokens = list(jieba.cut(query))
        bm25_scores = self._bm25.get_scores(query_tokens)
        top_bm25_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: self._top_k]
        bm25_docs = [self._docs[i] for i in top_bm25_idx]

        vector_docs = self._chroma.similarity_search(query, k=self._top_k)

        return reciprocal_rank_fusion(bm25_docs, vector_docs, top_n=top_n)


def load_retriever(
    persist_dir: str,
    collection_name: str,
    documents: list[Document],
    embedding_function,
    top_k: int = 20,
) -> HybridRetriever:
    store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_function,
    )
    return HybridRetriever(store, documents, top_k=top_k)
