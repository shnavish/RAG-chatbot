"""Interactive CLI (FR-18). Type 'quit' to exit (FR-19)."""
import sys
import time

from chain import MissingApiKeyError, stream_answer
from config import CHROMA_DIR
from interaction_log import log_answer
from retriever import get_merged_retriever, retrieve


def main() -> None:
    if not CHROMA_DIR.exists():
        print("Knowledge base not found. Run:  python ingest_all.py")
        sys.exit(1)

    print("Telecom Support Assistant (CLI). Type 'quit' to exit.\n")
    retriever = get_merged_retriever()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "quit":
            break

        start = time.perf_counter()
        docs = retrieve(question, retriever)
        print("Bot: ", end="", flush=True)
        parts = []
        try:
            for token in stream_answer(question, docs):
                parts.append(token)
                print(token, end="", flush=True)
        except MissingApiKeyError as e:
            print(e)
            continue
        print()
        print("Sources: " + ", ".join(d.metadata.get("ref", "?") for d in docs) + "\n")
        sources = [{"ref": d.metadata.get("ref", "?")} for d in docs]
        log_answer(question, "".join(parts), sources, time.perf_counter() - start, channel="cli")
    print("Goodbye!")


if __name__ == "__main__":
    main()
