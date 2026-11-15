"""Merged retriever over the three knowledge collections (FR-06, FR-07, FR-08).

To add a knowledge source (NFR-06): write an ingest_<name>.py that fills a Chroma
collection, then add one entry to COLLECTIONS below.
"""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from config import TOP_K, get_store

# collection name -> label used in the prompt context
COLLECTIONS = {
    "faq": "FAQ",
    "tickets": "TICKETS",
    "guides": "GUIDES",
}


def get_merged_retriever(k: int = TOP_K) -> RunnableParallel:
    """Query every collection in parallel; returns {collection: [Document, ...]}."""
    return RunnableParallel(
        {
            name: get_store(name).as_retriever(search_kwargs={"k": k})
            for name in COLLECTIONS
        }
    )


def retrieve(question: str, retriever: RunnableParallel | None = None) -> list[Document]:
    """Top-k from each collection, flattened in collection order (9 docs by default)."""
    retriever = retriever or get_merged_retriever()
    results = retriever.invoke(question)
    docs: list[Document] = []
    for name in COLLECTIONS:
        docs.extend(results.get(name, []))
    return docs


def format_context(docs: list[Document]) -> str:
    """Source-labelled context block for the prompt (FR-08)."""
    sections: dict[str, list[str]] = {label: [] for label in COLLECTIONS.values()}
    label_for = {"FAQ": "FAQ", "TICKET": "TICKETS", "GUIDE": "GUIDES"}
    for doc in docs:
        label = label_for.get(doc.metadata.get("source", ""), "OTHER")
        sections.setdefault(label, []).append(
            f"[{doc.metadata.get('ref', '?')}]\n{doc.page_content.strip()}"
        )
    blocks = []
    for label, items in sections.items():
        body = "\n\n".join(items) if items else "(no documents)"
        blocks.append(f"=== {label} ===\n{body}")
    return "\n\n".join(blocks)
