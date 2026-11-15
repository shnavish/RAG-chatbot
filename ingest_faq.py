"""Load data/faq.csv into the `faq` collection: one row = one document (FR-14)."""
import csv

from langchain_core.documents import Document

from config import FAQ_CSV, rebuild_collection

COLLECTION = "faq"


def load_docs():
    docs, ids = [], []
    with open(FAQ_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            faq_id = row["id"].strip()
            docs.append(
                Document(
                    page_content=f"Q: {row['question'].strip()}\nA: {row['answer'].strip()}",
                    metadata={
                        "source": "FAQ",
                        "ref": f"FAQ #{faq_id}",
                        "category": row.get("category", "").strip(),
                        "title": row["question"].strip(),
                    },
                )
            )
            ids.append(f"faq-{faq_id}")
    return docs, ids


def ingest():
    docs, ids = load_docs()
    n = rebuild_collection(COLLECTION, docs, ids)
    print(f"[faq] ingested {n} FAQ entries")
    return n


if __name__ == "__main__":
    ingest()
