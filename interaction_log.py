"""Append-only JSONL log of answers and thumbs feedback (FR-05a)."""
import json
import threading
import uuid
from datetime import datetime, timezone

from config import INTERACTION_LOG

_lock = threading.Lock()


def _write(record: dict) -> None:
    INTERACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with _lock, open(INTERACTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_answer(question: str, answer: str, sources: list[dict], latency_s: float,
               channel: str) -> str:
    interaction_id = uuid.uuid4().hex
    _write({
        "event": "answer",
        "id": interaction_id,
        "channel": channel,
        "question": question,
        "answer": answer,
        "sources": [s["ref"] for s in sources],
        "latency_s": round(latency_s, 3),
    })
    return interaction_id


def log_feedback(interaction_id: str, rating: str) -> None:
    """rating: 'up' or 'down'."""
    _write({"event": "feedback", "id": interaction_id, "rating": rating})
