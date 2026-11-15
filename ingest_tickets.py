"""Load resolved tickets from data/tickets.db into the `tickets` collection (FR-15)."""
import sqlite3

from langchain_core.documents import Document

from config import TICKETS_DB, rebuild_collection

COLLECTION = "tickets"


def load_docs():
    docs, ids = [], []
    con = sqlite3.connect(TICKETS_DB)
    try:
        rows = con.execute(
            "SELECT ticket_id, category, issue_type, description, resolution "
            "FROM tickets WHERE status = 'resolved' ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    for ticket_id, category, issue_type, description, resolution in rows:
        docs.append(
            Document(
                page_content=(
                    f"Issue: {issue_type}\nCustomer report: {description}\n"
                    f"Resolution: {resolution}"
                ),
                metadata={
                    "source": "TICKET",
                    "ref": ticket_id,
                    "category": category,
                    "title": issue_type,
                },
            )
        )
        ids.append(f"ticket-{ticket_id}")
    return docs, ids


def ingest():
    docs, ids = load_docs()
    n = rebuild_collection(COLLECTION, docs, ids)
    print(f"[tickets] ingested {n} resolved tickets")
    return n


if __name__ == "__main__":
    ingest()
