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
    find_submission,
    get_user_account,
    process_refund,
    recharge_phrase,
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
    choices: list[str] = field(default_factory=list)  # quick replies the UI shows as buttons


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


# A reply counts as a clear yes (or no) only if it contains a decisive word AND every other
# word is harmless filler. "sure, go ahead" and "yes please" pass; "ok", "ok thanks",
# "maybe", "yes but not now" and anything ending in "?" stay ambiguous (FR-36c, FR-36e).
_YES_STRONG = {"yes", "yeah", "yep", "yup", "sure", "proceed", "confirm", "confirmed", "y"}
_YES_PHRASES = ("go ahead", "do it", "go for it")
_YES_ALLOWED = _YES_STRONG | {"ok", "okay", "please", "go", "ahead", "do", "it", "for", "i", "refund",
                              "that", "thats", "fine", "thanks", "thank", "you", "good", "sounds",
                              "and", "the", "with", "process", "submit", "now"}
_NO_STRONG = {"no", "nope", "nah", "cancel", "stop", "dont", "not", "nevermind", "n"}
_NO_PHRASES = ("never mind",)
_NO_ALLOWED = _NO_STRONG | {"thanks", "thank", "you", "please", "it", "the", "refund", "proceed", "i",
                            "do", "want", "to", "need", "changed", "my", "mind", "never", "ok",
                            "okay", "that", "this", "go", "ahead", "with"}


def parse_confirmation(text: str) -> bool | None:
    """True = clear yes, False = clear no, None = anything else (ask again)."""
    if text.strip().endswith("?"):
        return None
    t = _norm(text)
    words = set(t.split())
    if not words:
        return None
    no_signal = bool(words & _NO_STRONG) or any(p in t for p in _NO_PHRASES)
    yes_signal = bool(words & _YES_STRONG) or any(p in t for p in _YES_PHRASES)
    if no_signal and words <= _NO_ALLOWED:  # "don't proceed": the negation wins (and refunds nothing)
        return False
    if yes_signal and not no_signal and words <= _YES_ALLOWED:
        return True
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
            text = (f"Your refund of {_fmt(result['amount'])} has been processed. "
                    f"Refund reference: **{result['refund_id']}**.\n\n"
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
            f"Just to check: your refund of {amount} is still waiting for your go-ahead, and "
            f"nothing has been refunded yet. Choose **Confirm refund** to go ahead, or **Cancel** "
            f"to drop it.",
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
        recharge = self._recharge(result["recharge_id"])
        if requested <= APPROVAL_LIMIT:  # FR-36a: limit enforced in code
            self.pending = {"amount": requested, "recharge_id": result["recharge_id"]}
            self.state = RefundState.AWAITING_CONFIRMATION
            step(trace, "Agent", "awaiting_user_confirmation")
            partial = ""
            if requested < result["recharge_amount"]:
                partial = (f" This closes the whole {_fmt(result['recharge_amount'])} recharge, so the "
                           "rest of it can't be refunded later.")
            return AgentReply(
                f"{offered_text}**Refund eligible: {_fmt(requested)}**\n\n"
                f"This refunds {recharge_phrase(recharge)}.{partial}\n\n"
                "Do you want to proceed with this refund?",
                self.state, needs_confirmation=True, trace=trace)

        # Above the limit: never processed here (FR-37, FR-39, GR-08).
        step(trace, "Agent", f"submit_to_support(amount={requested:g})")
        submission = submit_to_support(self.store, self.user_id, requested, result["recharge_id"])
        self.state = RefundState.REFUND_SUBMITTED_FOR_REVIEW
        if submission["already_submitted"]:  # one request per recharge, no duplicates
            step(trace, "Support", f"already submitted {submission['submission_id']}")
            return AgentReply(
                f"A refund request for {recharge_phrase(recharge)} is already with our Support Team "
                f"(request reference: {submission['submission_id']}), so there's no need to submit it "
                "again. They will review the request and process it or take the necessary action.",
                self.state, trace=trace)
        step(trace, "Support", f"submitted {submission['submission_id']}")
        return AgentReply(
            f"Your refund request for {_fmt(requested)} has been submitted to our Support Team "
            f"(request reference: {submission['submission_id']}). They will review the request "
            "and process it or take the necessary action.",
            self.state, trace=trace)

    def _recharge(self, recharge_id: str) -> dict:
        user = self.store.get_user(self.user_id)
        return next(r for r in user["recharges"] if r["recharge_id"] == recharge_id)

    def _no_amount(self, stated: list[float], trace: list[str]) -> AgentReply:
        if len(set(stated)) > 1:  # several numbers and the model didn't pick one
            self.state = RefundState.AWAITING_AMOUNT
            step(trace, "Agent", "awaiting_amount")
            return AgentReply("Which amount would you like refunded?", self.state, trace=trace)

        # Recharges already with the Support Team aren't offered again.
        eligible = [r for r in eligible_recharges(self.store, self.user_id)
                    if not find_submission(self.store, self.user_id, r["recharge_id"])]
        if not eligible:
            self.state = RefundState.REFUND_REJECTED
            step(trace, "Validation", "no eligible recharge")
            account = get_user_account(self.store, self.user_id)
            reasons = []
            for r in account["recharges"]:
                pending = find_submission(self.store, self.user_id, r["recharge_id"])
                reasons.append(
                    f"A refund request for {recharge_phrase(r)} is already with our Support Team "
                    f"(reference {pending['submission_id']})." if pending
                    else validate_refund(self.store, self.user_id, r["recharge_amount"])["reason"])
            listed = "\n".join(f"- {r}" for r in reasons) or "- There are no recharges on your account."
            return AgentReply("I couldn't find a recent recharge that can be refunded right now:\n\n" + listed,
                              self.state, trace=trace)

        if len(eligible) == 1:  # FR-42: offer the single eligible amount
            amount = float(eligible[0]["recharge_amount"])
            intro = f"I can see one recent recharge eligible for a refund: {recharge_phrase(eligible[0])}.\n\n"
            if amount <= APPROVAL_LIMIT:
                return self._validate_and_route(amount, trace, offered_text=intro)
            self.offer = amount
            self.state = RefundState.AWAITING_AMOUNT
            step(trace, "Agent", f"offered amount {amount:g}")
            return AgentReply(
                f"{intro}Refunds above {_fmt(APPROVAL_LIMIT)} are reviewed by our Support Team. "
                f"Would you like me to submit a refund request for {_fmt(amount)}?",
                self.state, trace=trace, choices=["Yes", "No"])

        # FR-42a: several eligible recharges -> ask which
        self.state = RefundState.AWAITING_AMOUNT
        step(trace, "Agent", "awaiting_amount")
        listed = "\n".join(f"- {recharge_phrase(r)[0].upper()}{recharge_phrase(r)[1:]}" for r in eligible)
        return AgentReply(f"I can see more than one recent recharge:\n\n{listed}\n\n"
                          "Which amount would you like refunded?", self.state, trace=trace,
                          choices=[_fmt(r["recharge_amount"]) for r in eligible])

    def _resolve_offer(self, accepted: bool, trace: list[str]) -> AgentReply:
        amount, self.offer = self.offer, None
        if not accepted:
            step(trace, "User", "declined offered amount")
            self.state = RefundState.REFUND_CANCELLED
            return AgentReply("Okay, I haven't submitted anything. No refund request was made.",
                              self.state, trace=trace)
        step(trace, "User", f"accepted offered amount {amount:g}")
        return self._validate_and_route(amount, trace)
