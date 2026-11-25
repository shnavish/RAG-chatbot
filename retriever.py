"""Merged retriever over the three knowledge collections (FR-06, FR-07, FR-08).

To add a knowledge source (NFR-06): write an ingest_<name>.py that fills a Chroma
collection, then add one entry to COLLECTIONS below.
"""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from config import TOP_K, get_embeddings, get_store

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


SOURCE_MIN_SCORE = 0.55  # answer-to-document similarity a source needs to be shown
SOURCE_MAX_GAP = 0.25  # ...and it must be within this much of the best-matching document
SOURCE_MAX_SHOWN = 5


def relevant_sources(answer: str, docs: list[Document]) -> list[Document]:
    """The retrieved documents the answer actually drew on, best first.

    All 9 retrieved documents go to the model (FR-07), but showing all 9 to the customer
    pads the list with near-misses. A document is shown when its text is similar to the
    final answer: a score of at least SOURCE_MIN_SCORE and within SOURCE_MAX_GAP of the
    best one. The best match is always kept.
    """
    if not docs or not answer.strip():
        return []
    emb = get_embeddings()  # normalised vectors, so the dot product is cosine similarity
    a = emb.embed_query(answer)
    scores = [sum(x * y for x, y in zip(a, d)) for d in emb.embed_documents([d.page_content for d in docs])]
    ranked = sorted(zip(scores, range(len(docs))), reverse=True)
    best = ranked[0][0]
    cutoff = max(SOURCE_MIN_SCORE, best - SOURCE_MAX_GAP)
    keep = [i for s, i in ranked if s >= cutoff] or [ranked[0][1]]
    return [docs[i] for i in keep[:SOURCE_MAX_SHOWN]]


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
