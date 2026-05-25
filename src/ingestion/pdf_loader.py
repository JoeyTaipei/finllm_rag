import re
from collections import defaultdict
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = Path(".chroma/finllm_kgi")
COLLECTION = "finllm_kgi"
ARTICLE_RE = re.compile(r"第\d+條(?:之\d+)?")

_splitter = RecursiveCharacterTextSplitter(
    separators=["\n第", "\n條", "\n項", "\n\n", "\n", "。"],
    chunk_size=500,
    chunk_overlap=100,
)


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path.stem, "page": page_num},
            ))
    return docs


def _split_and_tag(raw_docs: list[Document]) -> list[Document]:
    chunks = _splitter.split_documents(raw_docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_idx"] = idx
        m = ARTICLE_RE.search(chunk.page_content)
        chunk.metadata["article_number"] = m.group() if m else None
    return chunks


def load_and_index(
    pdf_dir: Path = Path("data/raw_pdfs"),
    chroma_dir: Path = CHROMA_DIR,
    collection: str = COLLECTION,
) -> tuple[list[Document], Chroma]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    all_chunks: list[Document] = []
    counts: dict[str, int] = defaultdict(int)

    for path in pdf_paths:
        raw_docs = _load_pdf(path)
        chunks = _split_and_tag(raw_docs)
        all_chunks.extend(chunks)
        counts[path.stem] += len(chunks)

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        all_chunks,
        embedding,
        persist_directory=str(chroma_dir),
        collection_name=collection,
    )

    print("=== Ingestion Summary ===")
    print(f"PDFs loaded : {len(pdf_paths)}")
    print(f"Total chunks: {len(all_chunks)}")
    print("\nChunks per document:")
    for stem, n in counts.items():
        print(f"  {stem[:50]:<52}: {n}")

    if all_chunks:
        s = all_chunks[0]
        print("\nSample chunk [0]:")
        print(f"  source      : {s.metadata['source']}")
        print(f"  page        : {s.metadata['page']}")
        print(f"  article     : {s.metadata['article_number']}")
        print(f"  chunk_idx   : {s.metadata['chunk_idx']}")
        print(f"  content     : {s.page_content[:120]}...")

    return all_chunks, store


if __name__ == "__main__":
    load_and_index()
