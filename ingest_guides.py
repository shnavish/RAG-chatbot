"""Chunk data/telecom_guide.pdf (600 chars, 100 overlap) into the `guides` collection (FR-16)."""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, GUIDE_PDF, rebuild_collection

COLLECTION = "guides"


# Repeated page header in the PDF ("Telecom Technical Reference Guide - Internal Use Only").
# It's not content, and customers shouldn't see it in source snippets.
HEADER_RE = re.compile(r"^\s*Telecom Technical Reference Guide\s*-\s*Internal Use Only\s*$", re.I | re.M)


def clean_page(text: str) -> str:
    return HEADER_RE.sub("", text or "").strip()


def load_docs():
    reader = PdfReader(str(GUIDE_PDF))
    pages = [
        Document(
            page_content=clean_page(page.extract_text()),
            metadata={"page": i + 1},
        )
        for i, page in enumerate(reader.pages)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents([p for p in pages if p.page_content])

    docs, ids = [], []
    for n, chunk in enumerate(chunks, start=1):
        page = chunk.metadata["page"]
        chunk.metadata = {
            "source": "GUIDE",
            "ref": f"{GUIDE_PDF.name} p.{page} chunk {n}",
            "page": page,
            "title": chunk.page_content.splitlines()[0][:80],
        }
        docs.append(chunk)
        ids.append(f"guide-{n}")
    return docs, ids


def ingest():
    docs, ids = load_docs()
    n = rebuild_collection(COLLECTION, docs, ids)
    print(f"[guides] ingested {n} guide chunks")
    return n


if __name__ == "__main__":
    ingest()
