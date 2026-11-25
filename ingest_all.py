"""Build every collection in chroma_store/. Safe to re-run (FR-17)."""
import ingest_faq
import ingest_guides
import ingest_tickets
from config import mark_kb_version


def ingest_all() -> int:
    total = ingest_faq.ingest() + ingest_tickets.ingest() + ingest_guides.ingest()
    mark_kb_version()
    return total


if __name__ == "__main__":
    print(f"Done: {ingest_all()} documents in chroma_store/")
