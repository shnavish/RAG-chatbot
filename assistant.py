"""Routes each message: intent detection -> existing RAG pipeline or the Refund Agent.

Used by both the Streamlit UI (app.py) and the CLI (main.py).
"""
from dataclasses import dataclass, field
from typing import Iterator

from chain import recent_history, stream_answer
from intent import REFUND_REQUEST, REQUIREMENT_INQUIRY, classify_intent
from refund_agent import RefundAgent, extract_amounts, parse_confirmation
from refund_data import DEMO_USER_ID, AccountStore
from retriever import retrieve
from workflow_log import step


@dataclass
class Turn:
    route: str  # "rag" or "refund"
    text: str | None = None  # refund replies are complete text
    stream: Iterator[str] | None = None  # RAG replies stream token by token
    docs: list = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    state: str | None = None  # refund workflow state after this turn (refund route only)


class Assistant:
    """One per chat session. Holds the simulated account data and refund state."""

    def __init__(self, store: AccountStore | None = None, user_id: str = DEMO_USER_ID,
                 retriever=None, agent: RefundAgent | None = None):
        self.store = store or AccountStore()
        self.agent = agent or RefundAgent(self.store, user_id)
        self.retriever = retriever

    def handle(self, message: str, history: list[dict]) -> Turn:
        """`history` = earlier {"role", "content"} messages, NOT including `message`."""
        history = recent_history(history)
        trace: list[str] = []

        # A refund is waiting for an explicit yes/no (FR-36c, FR-36e).
        if self.agent.awaiting_confirmation:
            decision = parse_confirmation(message)
            if decision is not None:
                return self._refund(self.agent.resolve_confirmation(decision, trace))
            intent = self._classify(message, history, trace)
            new_amounts = set(extract_amounts(message)) - {self.agent.pending["amount"]}
            if intent == REFUND_REQUEST and new_amounts:
                step(trace, "Agent", "pending refund replaced by a new request")
                self.agent.reset()
                return self._refund(self.agent.handle(message, history, trace))
            if intent == REQUIREMENT_INQUIRY and len(message.split()) >= 4:
                return self._rag(message, history, trace)  # refund stays pending
            return self._refund(self.agent.ask_again(trace))

        intent = self._classify(message, history, trace)
        if self.agent.awaiting_amount and (intent == REFUND_REQUEST or extract_amounts(message)
                                           or parse_confirmation(message) is not None):
            return self._refund(self.agent.handle(message, history, trace))
        if intent == REFUND_REQUEST:
            step(trace, "Route", "Refund Agent")
            return self._refund(self.agent.handle(message, history, trace))
        return self._rag(message, history, trace)

    def confirm(self, confirmed: bool) -> Turn:
        """Confirm Refund / Cancel buttons."""
        return self._refund(self.agent.resolve_confirmation(confirmed, []))

    # ----- helpers -----
    def _classify(self, message: str, history: list[dict], trace: list[str]) -> str:
        intent = classify_intent(message, history)
        step(trace, "Intent", intent)
        return intent

    def _rag(self, message: str, history: list[dict], trace: list[str]) -> Turn:
        step(trace, "Route", "RAG pipeline")
        docs = retrieve(message, self.retriever)  # raw question: history never changes retrieval
        return Turn(route="rag", stream=stream_answer(message, docs, history), docs=docs, trace=trace)

    def _refund(self, reply) -> Turn:
        return Turn(route="refund", text=reply.text, trace=reply.trace,
                    needs_confirmation=reply.needs_confirmation, state=reply.state.value)
