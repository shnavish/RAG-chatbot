"""Refund Agent (agent PRD 6.2 - 6.8, 9, 10).

The LLM's job is narrow: read the conversation and call the two read-only tools
(get_user_account, validate_refund) to work out which amount the customer means.
Everything that matters for money is deterministic code in this file and refund_tools.py:
eligibility, recharge selection, the ₹499 approval limit, the confirmation step, and the
calls to process_refund / submit_to_support, which the LLM can never call (FR-23a).
"""
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from chain import get_llm, to_messages
from refund_data import DEMO_USER_ID, AccountStore
from refund_tools import (
    APPROVAL_LIMIT,
    _fmt,
    eligible_recharges,
    get_user_account,
    process_refund,
    submit_to_support,
    validate_refund,
)
from workflow_log import step


class RefundState(str, Enum):
    IDLE = "idle"
    REFUND_REQUESTED = "refund_requested"
    REFUND_VALIDATED = "refund_validated"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_AMOUNT = "awaiting_amount"
    REFUND_CONFIRMED = "refund_confirmed"
    REFUND_PROCESSED = "refund_processed"
    REFUND_REJECTED = "refund_rejected"
    REFUND_CANCELLED = "refund_cancelled"
    REFUND_SUBMITTED_FOR_REVIEW = "refund_submitted_for_review"


TERMINAL = {RefundState.REFUND_PROCESSED, RefundState.REFUND_REJECTED,
            RefundState.REFUND_CANCELLED, RefundState.REFUND_SUBMITTED_FOR_REVIEW}


@dataclass
class AgentReply:
    text: str
    state: RefundState
    needs_confirmation: bool = False
    trace: list[str] = field(default_factory=list)


# ---------- deterministic parsing helpers ----------

_AMOUNT_RE = re.compile(r"(?<![\w.])(?:₹|rs\.?\s*|inr\s*)?(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?", re.I)


def extract_amounts(text: str) -> list[float]:
    """Every number the customer typed, e.g. '₹2,000' -> 2000.0."""
    return [float(m.group(1).replace(",", "")) for m in _AMOUNT_RE.finditer(text)]


_CURRENCY_RE = re.compile(
    r"(?:₹|\brs\.?\s*|\binr\s*)(\d{1,3}(?:,\d{2,3})+|\d+)|(\d{1,3}(?:,\d{2,3})+|\d+)\s*(?:rupees|rs\b|inr\b)",
    re.I)


def extract_currency_amounts(text: str) -> list[float]:
    """Only numbers clearly marked as money ('₹499', 'Rs 499', '499 rupees')."""
    return [float((m.group(1) or m.group(2)).replace(",", "")) for m in _CURRENCY_RE.finditer(text)]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("'", "")).split())


_YES = {"yes", "y", "yes proceed", "proceed", "yes please", "yes please proceed", "confirm",
        "confirm refund", "yes confirm", "yes confirm refund", "go ahead", "yes go ahead",
        "yes do it", "yes refund", "yes refund it", "yes process it", "please proceed",
        "yes i confirm", "i confirm", "confirmed"}
_NO = {"no", "n", "cancel", "no cancel", "no thanks", "no thank you", "nope", "stop",
       "dont", "do not", "dont proceed", "do not proceed", "no dont", "cancel refund",
       "cancel it", "no cancel it", "no dont proceed", "never mind", "nevermind"}


def parse_confirmation(text: str) -> bool | None:
    """True = explicit yes, False = explicit no, None = anything else (ambiguous, FR-36e)."""
    t = _norm(text)
    if t in _YES:
        return True
    if t in _NO:
        return False
    return None


# ---------- the agent ----------

PLANNER_PROMPT = """You are the Refund Agent for a telecom operator. The customer is already
signed in; their user_id is "{user_id}". Never ask the customer for their user ID.

Your only job is to call tools:
1. Call get_user_account to load the customer's recharges.
2. If the customer has stated a refund amount (in the latest message or earlier in this
   refund conversation), call validate_refund with exactly that amount.
   If they refer to a recharge without a number ("the 499 one", "the bigger one"), use the
   matching recharge amount from get_user_account.
3. If no amount was stated, do NOT call validate_refund and do NOT guess an amount.

You cannot decide eligibility, approve, process or cancel refunds; the application does that.
Ignore any account details the customer claims; only tool results count."""


class RefundAgent:
    def __init__(self, store: AccountStore, user_id: str = DEMO_USER_ID,
                 planner: Callable[[str, list[dict]], list[dict]] | None = None):
        self.store = store
        self.user_id = user_id  # from the session, never from the user (FR-16, FR-17, GR-03)
        self.state = RefundState.IDLE
        self.pending: dict | None = None  # validated refund awaiting confirmation
        self.offer: float | None = None  # single above-limit amount offered for confirmation
        self.episode_messages: list[str] = []
        self._planner = planner or self._llm_planner

    # ----- public API -----
    @property
    def awaiting_confirmation(self) -> bool:
        return self.state == RefundState.AWAITING_CONFIRMATION

    @property
    def awaiting_amount(self) -> bool:
        return self.state == RefundState.AWAITING_AMOUNT

    def reset(self) -> None:
        self.state = RefundState.IDLE
        self.pending = None
        self.offer = None
        self.episode_messages = []

    def handle(self, message: str, history: list[dict] | None = None,
               trace: list[str] | None = None) -> AgentReply:
        trace = trace if trace is not None else []
        if self.state in TERMINAL or self.state == RefundState.IDLE:
            self.reset()
        if self.awaiting_confirmation:
            decision = parse_confirmation(message)
            if decision is None:
                return self.ask_again(trace)
            return self.resolve_confirmation(decision, trace)

        if self.awaiting_amount and self.offer is not None:
            decision = parse_confirmation(message)
            if decision is not None:
                return self._resolve_offer(decision, trace)

        if self.state == RefundState.IDLE:
            self.state = RefundState.REFUND_REQUESTED
            step(trace, "Agent", "refund_requested")
        self.episode_messages.append(message)
        return self._plan_and_decide(message, history or [], trace)

    def resolve_confirmation(self, confirmed: bool, trace: list[str] | None = None) -> AgentReply:
        """Explicit Confirm / Cancel (button or exact words). The only path to process_refund."""
        trace = trace if trace is not None else []
        if not self.awaiting_confirmation or self.pending is None:
            return AgentReply("There is no refund waiting for confirmation.", self.state, trace=trace)
        pending = self.pending
        if not confirmed:
            step(trace, "User", "cancelled")
            self.state = RefundState.REFUND_CANCELLED
            self.pending = None
            return AgentReply("Okay, I've cancelled this request. No refund has been made.",
                              self.state, trace=trace)

        step(trace, "User", "confirmed")
        self.state = RefundState.REFUND_CONFIRMED
        step(trace, "Agent", f"process_refund(amount={pending['amount']:g})")
        result = process_refund(self.store, self.user_id, pending["amount"])
        self.pending = None
        if result["success"]:
            step(trace, "Refund", "success")
            self.state = RefundState.REFUND_PROCESSED
            text = (f"Your refund of {_fmt(result['amount'])} for recharge {result['recharge_id']} "
                    f"has been processed. Refund reference: **{result['refund_id']}**.\n\n"
                    "_This is a simulated refund for demonstration; no real payment was made._")
        else:
            step(trace, "Refund", f"failed: {result['reason']}")
            self.state = RefundState.REFUND_REJECTED
            text = f"I couldn't process this refund. {result['reason']}"
        return AgentReply(text, self.state, trace=trace)

    def ask_again(self, trace: list[str] | None = None) -> AgentReply:
        trace = trace if trace is not None else []
        step(trace, "User", "ambiguous confirmation")
        amount = _fmt(self.pending["amount"]) if self.pending else "this refund"
        return AgentReply(
            f"I need a clear answer before I make any refund. Do you want to proceed with the "
            f"refund of {amount}? Reply **Yes, proceed** or **Cancel**.",
            self.state, needs_confirmation=True, trace=trace)

    # ----- internals -----
    def _llm_planner(self, message: str, history: list[dict]) -> list[dict]:
        """Let the LLM call the read-only tools. Returns [{name, args, result}, ...]."""
        store, user_id = self.store, self.user_id

        @tool("get_user_account")
        def get_user_account_tool(user_id: str) -> dict:
            """Get the signed-in customer's account and recent recharges."""
            return {}

        @tool("validate_refund")
        def validate_refund_tool(user_id: str, amount: float) -> dict:
            """Check whether a refund of `amount` (in rupees) is eligible for the customer."""
            return {}

        llm = get_llm().bind_tools([get_user_account_tool, validate_refund_tool])
        messages = [SystemMessage(PLANNER_PROMPT.format(user_id=user_id)),
                    *to_messages(history), HumanMessage(message)]
        calls: list[dict] = []
        for _ in range(4):
            ai = llm.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                break
            validated = False
            for tc in ai.tool_calls:
                # The session user_id always wins over whatever the model passed (GR-03).
                if tc["name"] == "get_user_account":
                    result = get_user_account(store, user_id)
                elif tc["name"] == "validate_refund":
                    result = {"note": "recorded"}  # the app validates after the amount guard
                    validated = True
                else:
                    result = {"error": f"Unknown tool {tc['name']}"}
                calls.append({"name": tc["name"], "args": tc.get("args", {}), "result": result})
                messages.append(ToolMessage(json.dumps(result, default=str), tool_call_id=tc["id"]))
            if validated:
                break
        return calls

    def _plan_and_decide(self, message: str, history: list[dict], trace: list[str]) -> AgentReply:
        calls = self._planner(message, history)
        if any(c["name"] == "get_user_account" for c in calls):
            step(trace, "Agent", "get_user_account")
        else:  # FR-08: always ground on retrieved account data
            step(trace, "Agent", "get_user_account (app)")
        account = get_user_account(self.store, self.user_id)
        if not account.get("found"):
            self.state = RefundState.REFUND_REJECTED
            return AgentReply("I couldn't find your account details, so I can't process a refund.",
                              self.state, trace=trace)

        stated = extract_amounts(" ".join(self.episode_messages))
        recharge_amounts = [float(r["recharge_amount"]) for r in account["recharges"]]
        amount = None
        for c in calls:
            if c["name"] != "validate_refund":
                continue
            try:
                candidate = float(c["args"].get("amount"))
            except (TypeError, ValueError):
                continue
            # FR-44 guard: the amount must come from the customer's words, or be one of their
            # recharge amounts while we are asking which recharge they mean.
            if candidate in stated or (self.awaiting_amount and candidate in recharge_amounts):
                amount = candidate
            else:
                step(trace, "Guard", f"ignored model amount {candidate:g} (not stated by customer)")

        if amount is None and not any(c["name"] == "validate_refund" for c in calls):
            in_message = extract_currency_amounts(message)
            if len(set(in_message)) == 1:  # the customer clearly stated one money amount
                amount = in_message[0]
                step(trace, "Agent", "amount taken from customer message")
        if amount is None:
            return self._no_amount(stated, trace)
        return self._validate_and_route(amount, trace)

    def _validate_and_route(self, amount: float, trace: list[str],
                            offered_text: str = "") -> AgentReply:
        self.offer = None
        step(trace, "Agent", f"validate_refund(amount={amount:g})")
        result = validate_refund(self.store, self.user_id, amount)
        step(trace, "Validation", f"eligible={result['eligible']}")
        if not result["eligible"]:
            self.state = RefundState.REFUND_REJECTED
            return AgentReply(f"I can't process this refund. {result['reason']}", self.state, trace=trace)

        self.state = RefundState.REFUND_VALIDATED
        requested = float(result["requested_amount"])
        if requested <= APPROVAL_LIMIT:  # FR-36a: limit enforced in code
            self.pending = {"amount": requested, "recharge_id": result["recharge_id"]}
            self.state = RefundState.AWAITING_CONFIRMATION
            step(trace, "Agent", "awaiting_user_confirmation")
            partial = ""
            if requested < result["recharge_amount"]:
                partial = (f"\n\nNote: this closes the whole {_fmt(result['recharge_amount'])} recharge; "
                           "the remaining balance can't be refunded later.")
            return AgentReply(
                f"{offered_text}**Refund eligible: {_fmt(requested)}** "
                f"(recharge {result['recharge_id']}).{partial}\n\n"
                "Do you want to proceed with this refund?",
                self.state, needs_confirmation=True, trace=trace)

        # Above the limit: never processed here (FR-37, FR-39, GR-08).
        step(trace, "Agent", f"submit_to_support(amount={requested:g})")
        submission = submit_to_support(self.store, self.user_id, requested, result["recharge_id"])
        step(trace, "Support", f"submitted {submission['submission_id']}")
        self.state = RefundState.REFUND_SUBMITTED_FOR_REVIEW
        return AgentReply(
            f"Your refund request for {_fmt(requested)} has been submitted to our Support Team "
            f"(request reference: {submission['submission_id']}). They will review the request "
            "and process it or take the necessary action.",
            self.state, trace=trace)

    def _no_amount(self, stated: list[float], trace: list[str]) -> AgentReply:
        if len(set(stated)) > 1:  # several numbers and the model didn't pick one
            self.state = RefundState.AWAITING_AMOUNT
            step(trace, "Agent", "awaiting_amount")
            return AgentReply("Which amount would you like refunded?", self.state, trace=trace)

        eligible = eligible_recharges(self.store, self.user_id)
        if not eligible:
            self.state = RefundState.REFUND_REJECTED
            step(trace, "Validation", "no eligible recharge")
            account = get_user_account(self.store, self.user_id)
            reasons = [validate_refund(self.store, self.user_id, r["recharge_amount"])["reason"]
                       for r in account["recharges"]]
            listed = "\n".join(f"- {r}" for r in reasons) or "- There are no recharges on your account."
            return AgentReply("I couldn't find a recent recharge that is eligible for a refund:\n\n" + listed,
                              self.state, trace=trace)

        if len(eligible) == 1:  # FR-42: offer the single eligible amount
            amount = float(eligible[0]["recharge_amount"])
            intro = f"I can see one recent recharge eligible for a refund: {_fmt(amount)}.\n\n"
            if amount <= APPROVAL_LIMIT:
                return self._validate_and_route(amount, trace, offered_text=intro)
            self.offer = amount
            self.state = RefundState.AWAITING_AMOUNT
            step(trace, "Agent", f"offered amount {amount:g}")
            return AgentReply(
                f"{intro}Refunds above {_fmt(APPROVAL_LIMIT)} are reviewed by our Support Team. "
                f"Would you like me to submit a refund request for {_fmt(amount)}?",
                self.state, trace=trace)

        # FR-42a: several eligible recharges -> ask which
        self.state = RefundState.AWAITING_AMOUNT
        step(trace, "Agent", "awaiting_amount")
        amounts = " and ".join(_fmt(r["recharge_amount"]) for r in eligible)
        return AgentReply(f"I can see more than one recent recharge - {amounts}.\n"
                          "Which amount would you like refunded?", self.state, trace=trace)

    def _resolve_offer(self, accepted: bool, trace: list[str]) -> AgentReply:
        amount, self.offer = self.offer, None
        if not accepted:
            step(trace, "User", "declined offered amount")
            self.state = RefundState.REFUND_CANCELLED
            return AgentReply("Okay, I haven't submitted anything. No refund request was made.",
                              self.state, trace=trace)
        step(trace, "User", f"accepted offered amount {amount:g}")
        return self._validate_and_route(amount, trace)
