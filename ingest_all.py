"""Build every collection in chroma_store/. Safe to re-run (FR-17)."""
import ingest_faq
import ingest_guides
import ingest_tickets

if __name__ == "__main__":
    total = ingest_faq.ingest() + ingest_tickets.ingest() + ingest_guides.ingest()
    print(f"Done: {total} documents in chroma_store/")
