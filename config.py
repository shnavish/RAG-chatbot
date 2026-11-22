"""Shared settings and helpers for the telecom RAG chatbot."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Keep Chroma quiet and local before it is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # NFR-03: secrets come from .env, never from code

DATA_DIR = ROOT / "data"
FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

CHROMA_DIR = ROOT / "chroma_store"
LOG_DIR = ROOT / "logs"
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FR-09
# FR-13 names qwen/qwen3.6-27b, but Groq no longer serves it (404 model_not_found);
# qwen3.8-27b is the same family and size. Override with GROQ_MODEL in .env.
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
TOP_K = 3  # FR-07: top-3 per collection
HISTORY_EXCHANGES = 5  # agent PRD FR-50: recent user/assistant exchanges passed as context
CHUNK_SIZE = 600  # FR-16
CHUNK_OVERLAP = 100

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information in my knowledge base to answer that. "
    "Please call 611 or use the MyTelecom app to reach our support team."
)


@lru_cache(maxsize=1)
def get_embeddings():
    """Local MiniLM embeddings (NFR-02: no external embedding API)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_client():
    """One shared persistent Chroma client for the whole process."""
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def collection_counts(names) -> dict[str, int]:
    """Document count per collection, without loading the embedding model."""
    existing = {c.name if hasattr(c, "name") else c for c in get_client().list_collections()}
    return {
        n: get_client().get_collection(n).count() if n in existing else 0 for n in names
    }


def get_store(collection: str):
    """Open (or create) a persisted Chroma collection."""
    from langchain_chroma import Chroma

    return Chroma(
        client=get_client(),
        collection_name=collection,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def rebuild_collection(collection: str, docs, ids):
    """Idempotent re-ingest (FR-17): drop the collection, then add every doc with a stable id."""
    store = get_store(collection)
    store.delete_collection()
    store = get_store(collection)
    if docs:
        store.add_documents(documents=docs, ids=ids)
    return len(docs)
