"""Intent detection (agent PRD 6.1): requirement_inquiry or refund_request."""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from chain import get_llm, to_messages

REQUIREMENT_INQUIRY = "requirement_inquiry"
REFUND_REQUEST = "refund_request"


class IntentResult(BaseModel):
    """The customer's intent for the latest message."""

    intent: Literal["requirement_inquiry", "refund_request"] = Field(
        description="refund_request if the customer is asking us to refund, cancel or reverse "
        "one of their recharges (or is answering the refund assistant's question); otherwise "
        "requirement_inquiry."
    )


SYSTEM = """You classify the latest customer message for a telecom support assistant.

refund_request: the customer wants us to refund, cancel or reverse a recharge or payment of
theirs, OR they are replying to the refund assistant (giving an amount, saying yes/no).
requirement_inquiry: any question or problem to be answered from help content, including
questions ABOUT refunds or cancellation policy.

Examples:
"Why is my internet slow?" -> requirement_inquiry
"I want a refund of ₹999." -> refund_request
"Cancel my last recharge and give my money back." -> refund_request
"Please reverse the ₹499 recharge." -> refund_request
"How do I cancel a recharge?" -> requirement_inquiry
"What is your refund policy?" -> requirement_inquiry
"I want a refund." -> refund_request
(after the assistant asked which amount to refund) "the 499 one" -> refund_request

Use the earlier messages only to understand what the latest message refers to."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), MessagesPlaceholder("history"), ("human", "{message}")]
)

_REFUND_WORDS = ("refund", "money back", "reverse my", "reverse the", "cancel my recharge",
                 "cancel the recharge", "cancel my last recharge")


def _keyword_fallback(message: str) -> str:
    text = message.lower()
    asking_how = text.startswith(("how ", "what ", "can i", "is there", "do you"))
    if any(w in text for w in _REFUND_WORDS) and not asking_how:
        return REFUND_REQUEST
    return REQUIREMENT_INQUIRY


def classify_intent(message: str, history: list[dict] | None = None) -> str:
    """Return a structured intent value (FR-05) using the existing LLM config (FR-04)."""
    try:
        chain = PROMPT | get_llm().with_structured_output(IntentResult)
        result = chain.invoke({"message": message, "history": to_messages(history or [])})
        return result.intent
    except Exception:  # noqa: BLE001 - never block the user on a classifier failure
        return _keyword_fallback(message)
