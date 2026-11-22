"""Interactive CLI (FR-18). Type 'quit' to exit (FR-19).

Handles both support questions (RAG) and refund requests (Refund Agent). When a refund
needs confirmation, reply "Yes, proceed" or "Cancel". Type 'clear' to reset the conversation.
"""
import sys
import time

from assistant import Assistant
from chain import MissingApiKeyError
from config import CHROMA_DIR
from interaction_log import log_answer
from retriever import get_merged_retriever


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # "₹" on Windows consoles
    if not CHROMA_DIR.exists():
        print("Knowledge base not found. Run:  python ingest_all.py")
        sys.exit(1)

    print("Telecom Support Assistant (CLI). Type 'quit' to exit, 'clear' to start over.\n")
    retriever = get_merged_retriever()
    assistant = Assistant(retriever=retriever)
    history: list[dict] = []
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
        if question.lower() == "clear":  # resets history and any pending refund
            history = []
            assistant.agent.reset()
            print("(conversation cleared)\n")
            continue

        start = time.perf_counter()
        try:
            turn = assistant.handle(question, history)
            print("Bot: ", end="", flush=True)
            if turn.route == "rag":
                parts = []
                for token in turn.stream:
                    parts.append(token)
                    print(token, end="", flush=True)
                answer = "".join(parts)
                print()
                print("Sources: " + ", ".join(d.metadata.get("ref", "?") for d in turn.docs))
            else:
                answer = turn.text
                print(answer)
                if turn.needs_confirmation:
                    print('(Reply "Yes, proceed" or "Cancel")')
        except MissingApiKeyError as e:
            print(e)
            continue
        print()
        sources = [{"ref": d.metadata.get("ref", "?")} for d in turn.docs]
        log_answer(question, answer, sources, time.perf_counter() - start, channel="cli", route=turn.route)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    print("Goodbye!")


if __name__ == "__main__":
    main()
