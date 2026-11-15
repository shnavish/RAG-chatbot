"""Streamlit chat UI.  Run:  .venv\\Scripts\\streamlit run app.py"""
import time

import streamlit as st

from chain import MissingApiKeyError, stream_answer
from config import CHROMA_DIR, FALLBACK_MESSAGE, collection_counts
from interaction_log import log_answer, log_feedback
from retriever import COLLECTIONS, get_merged_retriever, retrieve

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


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ---------- sidebar ----------
with st.sidebar:
    st.header("📶 Support Assistant")
    st.caption("Answers come only from our FAQ, resolved support cases and user guides.")

    st.subheader("Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):  # FR-02
            st.session_state.pending = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):  # FR-04
        st.session_state.messages = []
        st.session_state.pending = None
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

if kb_empty:
    st.warning("The knowledge base is empty. Run `python ingest_all.py` once, then reload.")
    st.stop()

load_retriever()  # warm the embedding model once per server, with a visible spinner

for msg in st.session_state.messages:  # FR-03
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_feedback(msg)

typed = st.chat_input("Ask a question about your mobile service…")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        sources: list[dict] = []
        try:
            with st.spinner("Searching the knowledge base…"):
                docs = retrieve(question, load_retriever())
            sources = to_sources(docs)
            answer = st.write_stream(stream_answer(question, docs))  # FR-05
        except MissingApiKeyError as e:
            answer = f"⚠️ {e}"
            st.markdown(answer)
        except Exception as e:  # noqa: BLE001 - network / API failures
            answer = f"⚠️ Sorry, something went wrong reaching the assistant. {FALLBACK_MESSAGE}"
            st.markdown(answer)
            st.caption(f"Error: {type(e).__name__}: {e}")

        latency = time.perf_counter() - start
        answer = answer if isinstance(answer, str) else "".join(answer)
        msg = {"role": "assistant", "content": answer, "sources": sources}
        if sources and not answer.startswith("⚠️"):
            msg["id"] = log_answer(question, answer, sources, latency, channel="web")
        render_sources(sources)
        render_feedback(msg)
        st.session_state.messages.append(msg)
