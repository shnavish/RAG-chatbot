"""Streamlit chat UI.  Run:  .venv\\Scripts\\streamlit run app.py"""
import time

import streamlit as st

from assistant import Assistant
from chain import MissingApiKeyError
from config import CHROMA_DIR, FALLBACK_MESSAGE, collection_counts
from interaction_log import log_answer, log_feedback
from refund_data import AccountStore
from retriever import COLLECTIONS, get_merged_retriever

REFUND_SAMPLES = [
    "I want a refund of ₹499.",
    "I want a refund of ₹999.",
    "Give me a refund of ₹2,000.",
    "I want a refund.",
]

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "I have an unexpected charge on my bill. What is it?",
    "How do I activate roaming before travelling abroad?",
    "My eSIM won't activate. What should I do?",
    "My phone says SIM not recognised.",
    "There is an echo on my calls.",
    "I can't log in to the MyTelecom app.",
    "How do I download my itemised bill?",
]

SOURCE_BADGE = {"FAQ": "📘 FAQ", "TICKET": "🎫 TICKET", "GUIDE": "📄 GUIDE"}

st.set_page_config(page_title="Telecom Support Assistant", page_icon="📶", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_retriever():
    return get_merged_retriever()


def kb_counts() -> dict[str, int]:
    return collection_counts(COLLECTIONS)


def to_sources(docs) -> list[dict]:
    return [
        {
            "source": d.metadata.get("source", "?"),
            "ref": d.metadata.get("ref", "?"),
            "title": d.metadata.get("title", ""),
            "snippet": d.page_content.strip(),
        }
        for d in docs
    ]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{SOURCE_BADGE.get(s['source'], s['source'])} · {s['ref']}** — {s['title']}")
            st.caption(s["snippet"][:400] + ("…" if len(s["snippet"]) > 400 else ""))


def on_feedback(key: str, interaction_id: str) -> None:
    value = st.session_state.get(key)
    if value is not None:
        log_feedback(interaction_id, "up" if value == 1 else "down")


def render_feedback(msg: dict) -> None:
    if msg.get("id"):
        key = f"fb_{msg['id']}"
        st.feedback("thumbs", key=key, on_change=on_feedback, args=(key, msg["id"]))


def render_trace(trace: list[str]) -> None:
    if trace:
        with st.expander("Workflow steps"):
            st.code("\n".join(trace), language=None)


def set_action(action: str) -> None:
    st.session_state.pending_action = action


def render_confirm_buttons(index: int) -> None:
    """Refund confirmation step (agent PRD section 11)."""
    left, right, _ = st.columns([2, 1.2, 2])
    left.button("Confirm Refund", key=f"confirm_{index}", type="primary", use_container_width=True,
                on_click=set_action, args=("confirm",))
    right.button("Cancel", key=f"cancel_{index}", use_container_width=True,
                 on_click=set_action, args=("cancel",))


# ---------- session state ----------
for key, default in {"messages": [], "pending": None, "pending_action": None,
                     "store": None, "assistant": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.store is None:
    st.session_state.store = AccountStore()


def new_assistant() -> Assistant:
    return Assistant(store=st.session_state.store, retriever=load_retriever())


# ---------- sidebar ----------
with st.sidebar:
    st.header("📶 Support Assistant")
    st.caption("Answers come only from our FAQ, resolved support cases and user guides.")

    st.subheader("Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):  # FR-02
            st.session_state.pending = q

    st.subheader("Try a refund")
    for q in REFUND_SAMPLES:
        if st.button(q, use_container_width=True, key=f"refund_{q}"):
            st.session_state.pending = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):  # FR-04, agent FR-48
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.pending_action = None
        st.session_state.assistant = None  # drops any pending refund
        st.rerun()

    with st.expander("Demo account (simulated)"):
        user = st.session_state.store.get_user("user_123")
        st.caption(f"Signed in as {user['name']} · {user['plan']} plan")
        for r in user["recharges"]:
            st.write(f"{r['recharge_id']}: ₹{r['recharge_amount']} · {r['recharge_age_days']}d ago · "
                     f"{r['refund_status']}")
        if st.button("Reset demo account", use_container_width=True):
            st.session_state.store = AccountStore()
            st.session_state.assistant = None
            st.rerun()

    with st.expander("Knowledge base"):
        try:
            counts = kb_counts()
            for name, n in counts.items():
                st.write(f"{COLLECTIONS[name]}: {n} documents")
        except Exception as e:  # noqa: BLE001
            counts = {}
            st.error(f"Could not open {CHROMA_DIR.name}/: {e}")

kb_empty = not any(counts.values())

# ---------- main ----------
st.title("How can we help today?")
st.caption(
    "I can help with mobile data, billing questions, SIM/eSIM, roaming, calls and the "
    "MyTelecom app. I can't see your account, balance or bill — for those, use the "
    "MyTelecom app or call 611."
)

@st.cache_resource(show_spinner="Building the knowledge base (first start only)…")
def build_knowledge_base() -> int:
    """Hosted copies start without chroma_store/ (it isn't in git), so build it once."""
    import ingest_faq
    import ingest_guides
    import ingest_tickets

    return ingest_faq.ingest() + ingest_tickets.ingest() + ingest_guides.ingest()


if kb_empty:
    try:
        build_knowledge_base()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build the knowledge base: {e}. Run `python ingest_all.py`, then reload.")
        st.stop()
    if not any(kb_counts().values()):
        st.error("The knowledge base is still empty after building. Run `python ingest_all.py`, then reload.")
        st.stop()
    st.rerun()

load_retriever()  # warm the embedding model once per server, with a visible spinner
if st.session_state.assistant is None:
    st.session_state.assistant = new_assistant()
assistant: Assistant = st.session_state.assistant

messages = st.session_state.messages
for i, msg in enumerate(messages):  # FR-03
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_trace(msg.get("trace", []))
            if (msg.get("confirm") and i == len(messages) - 1
                    and assistant.agent.awaiting_confirmation and not st.session_state.pending_action):
                render_confirm_buttons(i)
            render_feedback(msg)

typed = st.chat_input("Ask a question or request a refund…")
action = st.session_state.pending_action
question = typed or st.session_state.pending
st.session_state.pending = None
st.session_state.pending_action = None

if action:  # Confirm Refund / Cancel button
    question = "Yes, proceed" if action == "confirm" else "Cancel"

if question:
    history = list(messages)  # earlier messages, before this one (agent FR-46)
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        sources: list[dict] = []
        trace: list[str] = []
        turn = None
        try:
            with st.spinner("Thinking…"):
                turn = assistant.confirm(action == "confirm") if action else assistant.handle(question, history)
            trace = turn.trace
            if turn.route == "rag":
                sources = to_sources(turn.docs)
                answer = st.write_stream(turn.stream)  # FR-05
            else:
                answer = turn.text
                st.markdown(answer)
        except MissingApiKeyError as e:
            answer = f"⚠️ {e}"
            st.markdown(answer)
        except Exception as e:  # noqa: BLE001 - network / API failures
            answer = f"⚠️ Sorry, something went wrong reaching the assistant. {FALLBACK_MESSAGE}"
            st.markdown(answer)
            st.caption(f"Error: {type(e).__name__}: {e}")

        latency = time.perf_counter() - start
        answer = answer if isinstance(answer, str) else "".join(answer)
        msg = {"role": "assistant", "content": answer, "sources": sources, "trace": trace,
               "confirm": bool(turn and turn.needs_confirmation)}
        if turn and not answer.startswith("⚠️"):
            msg["id"] = log_answer(question, answer, sources, latency, channel="web", route=turn.route)
        render_sources(sources)
        render_trace(trace)
        messages.append(msg)
        if turn and turn.route == "refund":
            st.rerun()  # redraw so the confirm buttons and demo-account panel reflect the new state
        render_feedback(msg)
