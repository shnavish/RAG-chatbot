"""Prompt + Groq LLM chain (FR-10 .. FR-13)."""
import os
from functools import lru_cache
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import FALLBACK_MESSAGE, LLM_MODEL
from retriever import format_context, get_merged_retriever, retrieve

SYSTEM_PROMPT = """You are NovaCell's customer care assistant for a mobile telecom operator.
You help customers with mobile data, connectivity, billing, SIM/eSIM, roaming, voice calls,
devices and account/app questions.

STRICT RULES
1. Answer ONLY from the CONTEXT below. Do not use outside knowledge. Never invent prices,
   fees, speeds, time limits, policies, phone numbers, URLs or steps that are not in the CONTEXT.
2. If NOTHING in the CONTEXT (FAQ, TICKETS or GUIDES) relates to the customer's issue, reply
   with exactly:
   "{fallback}"
   If any document is relevant, answer from it instead of using this fallback.
3. You cannot see the customer's account, balance, bill, usage or plan. If they ask about
   their own account data, explain the general rule from the CONTEXT (if any) and point them
   to the self-serve option or support channel named in the CONTEXT.
4. TICKETS are past resolved cases of the same kind of issue. When a ticket matches, use it:
   explain the likely cause it shows, give any steps the customer can take themselves, and say
   that support can apply the fix (via 611 or the MyTelecom app) if it needs action on their
   account or the network. Never promise the same refund, credit, compensation or outcome a
   past customer received.
5. Be concise and friendly. Use short numbered steps for troubleshooting.
6. Do not mention "the context", "documents" or source labels in your answer; the app shows
   sources separately.

CONTEXT
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
).partial(fallback=FALLBACK_MESSAGE)


class MissingApiKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise MissingApiKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,  # FR-12
        reasoning_effort="none",  # FR-13
        streaming=True,
    )


def get_answer_chain():
    """prompt | llm | parser. Input: {"context": str, "question": str}."""
    return PROMPT | get_llm() | StrOutputParser()


def get_rag_chain():
    """Full LCEL chain: question -> merged retrieval -> prompt -> LLM -> text."""
    retriever = get_merged_retriever()
    to_context = RunnableLambda(lambda q: format_context(retrieve(q, retriever)))
    return (
        {"context": to_context, "question": RunnablePassthrough()}
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )


def stream_answer(question: str, docs: list[Document]) -> Iterator[str]:
    """Stream the answer token by token (FR-05) for already-retrieved docs."""
    return get_answer_chain().stream(
        {"context": format_context(docs), "question": question}
    )


def ask(question: str) -> tuple[list[Document], Iterator[str]]:
    """Retrieve, then return (docs, token stream). Used by the CLI and the UI."""
    docs = retrieve(question)
    return docs, stream_answer(question, docs)
