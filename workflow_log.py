"""Workflow step logging (agent PRD section 12). Logs steps only, never model reasoning."""
import logging

from config import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("workflow")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False


def step(trace: list[str], tag: str, message: str) -> None:
    """Record one workflow step, e.g. step(trace, "Intent", "refund_request")."""
    line = f"[{tag}] {message}"
    trace.append(line)
    logger.info(line)
