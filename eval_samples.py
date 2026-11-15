"""Check the PRD goal: >70% of sample questions answered without a "call 611" fallback.

Usage:  python eval_samples.py            (retrieval + generation, needs GROQ_API_KEY)
        python eval_samples.py --retrieval (retrieval only, no API calls)
"""
import sys
import time

from chain import stream_answer  # fallback text is config.FALLBACK_MESSAGE
from retriever import get_merged_retriever, retrieve

SAMPLES = [
    "Why is my mobile internet so slow?",
    "My internet stopped working completely.",
    "My signal keeps dropping.",
    "Is there a network outage in my area?",
    "I got unexpected roaming charges.",
    "I have no service abroad.",
    "My phone says SIM not recognised.",
    "My eSIM activation failed.",
    "My number porting is delayed.",
    "I was billed twice this month.",
    "My plan auto-renewed at the wrong price.",
    "I can't download my itemised bill.",
    "Calls to some numbers are barred.",
    "There is an echo on my calls.",
    "My phone doesn't support VoLTE.",
    "I can't log in to the MyTelecom app.",
    "Someone changed my plan without permission.",
    "How do I activate roaming before a trip?",
    "What is the capital of France?",  # out of scope: should fall back
    "What is my current balance?",  # personal data: should redirect
]


def main() -> None:
    retriever = get_merged_retriever()
    retrieval_only = "--retrieval" in sys.argv
    answered = 0
    for i, q in enumerate(SAMPLES):
        if i and not retrieval_only:
            time.sleep(4)  # stay under Groq free-tier rate limits so latency reflects one user
        start = time.perf_counter()
        docs = retrieve(q, retriever)
        refs = ", ".join(d.metadata["ref"] for d in docs[:3])
        if retrieval_only:
            print(f"- {q}\n    top FAQ: {refs}")
            continue
        first = None
        parts = []
        for tok in stream_answer(q, docs):
            first = first or time.perf_counter() - start
            parts.append(tok)
        answer = "".join(parts)
        total = time.perf_counter() - start
        fallback = "don't have enough information" in answer
        answered += not fallback
        flag = "FALLBACK" if fallback else "ANSWERED"
        print(f"[{flag}] {q}  (first token {first or 0:.1f}s, total {total:.1f}s)")
        print("    " + answer.replace("\n", "\n    ")[:600] + "\n")
    if not retrieval_only:
        print(f"Answered without 611: {answered}/{len(SAMPLES)} = {answered / len(SAMPLES):.0%}")


if __name__ == "__main__":
    main()
