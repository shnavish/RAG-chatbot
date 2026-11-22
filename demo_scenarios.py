"""Run the seven demonstration scenarios from agent_PRD.md against the live model.

Usage:  .venv\\Scripts\\python demo_scenarios.py
Needs GROQ_API_KEY. Each scenario starts from a fresh session and its own account data.
"""
import copy
import sys
import time
from unittest import mock

import refund_agent
from assistant import Assistant
from refund_data import USERS, AccountStore
from refund_tools import process_refund
from retriever import get_merged_retriever

RETRIEVER = get_merged_retriever()
results: list[tuple[str, bool, str]] = []


def session(**recharge_changes) -> Assistant:
    users = copy.deepcopy(USERS)
    for rid, fields in recharge_changes.items():
        for r in users["user_123"]["recharges"]:
            if r["recharge_id"] == rid:
                r.update(fields)
    return Assistant(store=AccountStore(users), retriever=RETRIEVER)


def say(assistant: Assistant, history: list[dict], text: str):
    time.sleep(3)  # stay under Groq free-tier rate limits
    turn = assistant.handle(text, history)
    answer = "".join(turn.stream) if turn.route == "rag" else turn.text
    history += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
    print(f"  User: {text}\n  Bot ({turn.route}): {answer.strip()[:400]}")
    for line in turn.trace:
        print(f"      {line}")
    return turn, answer


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  => {'PASS' if ok else 'FAIL'} {detail}\n")


def run():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with mock.patch.object(refund_agent, "process_refund", wraps=process_refund) as spy:
        print("Scenario 1 - successful refund (₹499)")
        a, h = session(), []
        t1, _ = say(a, h, "I want a refund of ₹499.")
        asked = t1.route == "refund" and t1.needs_confirmation and spy.call_count == 0
        t2, ans = say(a, h, "Yes, proceed")
        status = a.store.get_user("user_123")["recharges"][1]["refund_status"]
        check("1 successful refund", asked and spy.call_count == 1 and status == "refunded" and "ref_" in ans,
              f"confirmation asked={asked}, process_refund calls={spy.call_count}, status={status}")

        print("Scenario 2 - amount exceeds recharge (₹2,000)")
        spy.reset_mock()
        a, h = session(), []
        _, ans = say(a, h, "Give me a refund of ₹2,000.")
        check("2 exceeds recharge", spy.call_count == 0 and "exceeds" in ans,
              f"process_refund calls={spy.call_count}")

        print("Scenario 3 - recharge too old (15 days)")
        spy.reset_mock()
        a, h = session(rch_457={"recharge_age_days": 15}), []
        _, ans = say(a, h, "I want a refund of ₹499.")
        check("3 too old", spy.call_count == 0 and "7-day" in ans, f"process_refund calls={spy.call_count}")

        print("Scenario 4 - already refunded")
        spy.reset_mock()
        a, h = session(rch_457={"refund_status": "refunded"}), []
        _, ans = say(a, h, "I want a refund of ₹499.")
        check("4 already refunded", spy.call_count == 0 and "already been refunded" in ans,
              f"process_refund calls={spy.call_count}")

        print("Scenario 5 - existing RAG query")
        a, h = session(), []
        t, _ = say(a, h, "Why is my mobile internet slow?")
        check("5 RAG query", t.route == "rag" and len(t.docs) == 9, f"route={t.route}, docs={len(t.docs)}")

        print("Scenario 6 - above approval limit (₹999)")
        spy.reset_mock()
        a, h = session(), []
        t, ans = say(a, h, "I want a refund of ₹999.")
        status = a.store.get_user("user_123")["recharges"][0]["refund_status"]
        check("6 submitted to support",
              spy.call_count == 0 and not t.needs_confirmation and "Support Team" in ans
              and status == "not_refunded" and len(a.store.support_submissions) == 1,
              f"process_refund calls={spy.call_count}, status={status}")

        print("Scenario 7 - follow-up question uses history")
        a, h = session(), []
        say(a, h, "How do I activate international roaming?")
        t, ans = say(a, h, "Should I do that before I travel?")
        check("7 follow-up", t.route == "rag" and "24 hours" in ans,
              f"route={t.route}, mentions 24 hours={'24 hours' in ans}")

        print("Extra - missing amount, then choose one")
        spy.reset_mock()
        a, h = session(), []
        t1, ans1 = say(a, h, "I want a refund.")
        t2, _ = say(a, h, "The 499 one")
        check("missing amount", "₹999 and ₹499" in ans1 and t2.needs_confirmation and spy.call_count == 0,
              f"asked which={'₹999 and ₹499' in ans1}, then confirmation={t2.needs_confirmation}")

        print("Extra - question about cancelling is an inquiry (FR-05a)")
        a, h = session(), []
        t, _ = say(a, h, "How do I cancel a recharge?")
        check("how-to cancel is inquiry", t.route == "rag", f"route={t.route}")

    print("Summary")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")


if __name__ == "__main__":
    run()
