"""Streamlit chat UI.  Run:  .venv\\Scripts\\streamlit run app.py"""
import html
import logging
import time

import streamlit as st

from assistant import Assistant
from chain import MissingApiKeyError
from config import FALLBACK_MESSAGE, collection_counts
from interaction_log import log_answer, log_feedback
from refund_data import AccountStore
from retriever import COLLECTIONS, get_merged_retriever

BRAND = "NovaCell Support"

# Welcome-screen topic cards: (icon, title, description, questions)
TOPICS = [
    (":material/network_check:", "Internet & signal", "Slow data, dropped signal, outages",
     ["Why is my mobile internet so slow?", "My signal keeps dropping."]),
    (":material/receipt_long:", "Billing & charges", "Unexpected charges, itemised bills",
     ["I have an unexpected charge on my bill. What is it?", "How do I download my itemised bill?"]),
    (":material/sim_card:", "SIM, eSIM & roaming", "Activation and travel setup",
     ["My eSIM won't activate. What should I do?", "How do I activate roaming before travelling abroad?"]),
    (":material/currency_rupee:", "Refunds", "Refund a recent recharge",
     ["I want a refund of ₹499.", "I want a refund."]),
]

SAMPLE_QUESTIONS = [  # sidebar (FR-02)
    "Why is my mobile internet so slow?",
    "I have an unexpected charge on my bill. What is it?",
    "How do I activate roaming before travelling abroad?",
    "My phone says SIM not recognised.",
    "There is an echo on my calls.",
    "I can't log in to the MyTelecom app.",
]

REFUND_SAMPLES = [
    "I want a refund of ₹499.",
    "I want a refund of ₹999.",
    "Give me a refund of ₹2,000.",
    "I want a refund.",
]

SOURCE_STYLE = {  # source type -> (label, badge colour, icon)
    "FAQ": ("FAQ", "blue", ":material/help:"),
    "TICKET": ("Past case", "orange", ":material/confirmation_number:"),
    "GUIDE": ("Guide", "violet", ":material/menu_book:"),
}

REFUND_STATUS = {  # refund state -> (label, badge colour, icon)
    "awaiting_confirmation": ("Confirmation needed", "blue", ":material/pending_actions:"),
    "awaiting_amount": ("More details needed", "orange", ":material/help:"),
    "refund_processed": ("Refund processed", "green", ":material/check_circle:"),
    "refund_submitted_for_review": ("Sent to Support Team", "violet", ":material/support_agent:"),
    "refund_rejected": ("Not eligible", "red", ":material/block:"),
    "refund_cancelled": ("Cancelled", "gray", ":material/cancel:"),
}

AVATARS = {"user": ":material/person:", "assistant": ":material/support_agent:"}

log = logging.getLogger("app")

st.set_page_config(page_title=BRAND, page_icon=":material/support_agent:", layout="centered",
                   initial_sidebar_state="auto")  # open on desktop, collapsed on phones

# Small, scoped styling on top of the theme in .streamlit/config.toml.
st.html("""
<style>
.nc-header {display:flex; align-items:center; gap:.75rem; padding:.25rem 0 .75rem;}
.nc-logo {width:42px; height:42px; border-radius:12px; flex:none; display:flex; align-items:flex-end;
          justify-content:center; gap:3px; padding-bottom:11px; box-sizing:border-box;
          background:linear-gradient(135deg,#3B5BDB,#5C7CFA);}
.nc-logo span {width:5px; border-radius:2px; background:#fff; display:block;}
.nc-logo .b1 {height:7px; opacity:.75;} .nc-logo .b2 {height:12px; opacity:.9;} .nc-logo .b3 {height:18px;}
.nc-side-brand .nc-logo {padding-bottom:8px; gap:2px;}
.nc-side-brand .nc-logo span {width:4px;}
.nc-side-brand .nc-logo .b1 {height:5px;} .nc-side-brand .nc-logo .b2 {height:9px;} .nc-side-brand .nc-logo .b3 {height:14px;}
/* question links: left-aligned, wrap neatly */
[data-testid="stBaseButton-tertiary"] {justify-content:flex-start; text-align:left;}
[data-testid="stBaseButton-tertiary"] > div {justify-content:flex-start;}
[data-testid="stBaseButton-tertiary"] p {text-align:left;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.45rem;}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {min-height:0; padding:.2rem 0;}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p {font-size:.9rem; line-height:1.35;}
.nc-name {font-weight:700; font-size:1.15rem; line-height:1.2;}
.nc-tag {font-size:.85rem; opacity:.65;}
.nc-pill {margin-left:auto; display:inline-flex; align-items:center; gap:.4rem; white-space:nowrap;
          font-size:.75rem; font-weight:600; padding:.25rem .65rem; border-radius:999px;
          background:rgba(16,185,129,.12); color:#0F9F6E;}
.nc-dot {width:7px; height:7px; border-radius:50%; background:#10B981;}
.nc-hero {padding:1.25rem 0 .5rem;}
.nc-hero h2 {font-size:1.9rem; font-weight:700; margin:0 0 .35rem; padding:0;}
.nc-hero p {opacity:.7; margin:0; font-size:1rem;}
.nc-side-brand {display:flex; align-items:center; gap:.6rem; margin-bottom:.25rem;}
.nc-side-brand .nc-logo {width:34px; height:34px; border-radius:10px;}
.nc-label {font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
           opacity:.55; margin:.5rem 0 .1rem;}
[data-testid="stChatMessage"] {padding:.85rem 1rem;}
@media (max-width: 640px) { .nc-pill {display:none;} .nc-hero h2 {font-size:1.5rem;} }
</style>
""")

# Signal-bars mark drawn in CSS (st.html strips inline SVG).
LOGO_SVG = '<span class="b1"></span><span class="b2"></span><span class="b3"></span>'


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_retriever():
    return get_merged_retriever()


@st.cache_resource(show_spinner="Building the knowledge base (first start only)…")
def build_knowledge_base() -> int:
    """Hosted copies start without chroma_store/ (it isn't in git), so build it once."""
    import ingest_faq
    import ingest_guides
    import ingest_tickets

    return ingest_faq.ingest() + ingest_tickets.ingest() + ingest_guides.ingest()


def kb_counts() -> dict[str, int]:
    return collection_counts(COLLECTIONS)


def to_sources(docs) -> list[dict]:
    sources = []
    for d in docs:
        meta = d.metadata
        title = meta.get("title", "")
        if meta.get("source") == "GUIDE":  # chunk text starts mid-sentence; name the page instead
            title = f"Telecom user guide · page {meta.get('page', '?')}"
        sources.append({"source": meta.get("source", "?"), "ref": meta.get("ref", "?"),
                        "title": title, "snippet": d.page_content.strip()})
    return sources


# ---------- rendering helpers ----------

def ask(question: str) -> None:
    st.session_state.pending = question


def set_action(action: str) -> None:
    st.session_state.pending_action = action


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)}", icon=":material/library_books:"):
        for s in sources:
            label, color, icon = SOURCE_STYLE.get(s["source"], (s["source"], "gray", ":material/description:"))
            title = s["title"] or s["ref"]
            st.markdown(f":{color}-badge[{icon} {label}] **{title}**  \n"
                        f"<span style='opacity:.55;font-size:.8rem'>{html.escape(s['ref'])}</span>",
                        unsafe_allow_html=True)
            snippet = s["snippet"].replace("\n", " ")
            st.caption(snippet[:260] + ("…" if len(snippet) > 260 else ""))


def render_trace(trace: list[str]) -> None:
    if trace:
        with st.expander("How this was handled", icon=":material/account_tree:"):
            st.code("\n".join(trace), language=None)


def on_feedback(key: str, interaction_id: str) -> None:
    value = st.session_state.get(key)
    if value is not None:
        log_feedback(interaction_id, "up" if value == 1 else "down")


def render_feedback(msg: dict) -> None:
    if msg.get("id"):
        key = f"fb_{msg['id']}"
        st.feedback("thumbs", key=key, on_change=on_feedback, args=(key, msg["id"]))


def render_confirm_buttons(index: int) -> None:
    """Refund confirmation step (agent PRD section 11)."""
    with st.container(horizontal=True):
        st.button("Confirm refund", key=f"confirm_{index}", type="primary", icon=":material/check:",
                  on_click=set_action, args=("confirm",))
        st.button("Cancel", key=f"cancel_{index}", icon=":material/close:",
                  on_click=set_action, args=("cancel",))


def render_refund_card(msg: dict, show_buttons: bool, index: int) -> None:
    with st.container(border=True):
        status = REFUND_STATUS.get(msg.get("refund_state") or "")
        if status:
            label, color, icon = status
            st.markdown(f":{color}-badge[{icon} {label}]")
        st.markdown(msg["content"])
        if show_buttons:
            render_confirm_buttons(index)


def render_assistant_body(msg: dict, show_buttons: bool, index: int) -> None:
    if msg.get("error"):
        st.warning(msg["content"].removeprefix("⚠️").strip(), icon=":material/error:")
    elif msg.get("route") == "refund":
        render_refund_card(msg, show_buttons, index)
    else:
        st.markdown(msg["content"])
    render_sources(msg.get("sources", []))
    render_trace(msg.get("trace", []))
    render_feedback(msg)


def header() -> None:
    st.html(f"""
    <div class="nc-header">
      <div class="nc-logo">{LOGO_SVG}</div>
      <div><div class="nc-name">{BRAND}</div>
           <div class="nc-tag">Customer care assistant</div></div>
      <span class="nc-pill"><span class="nc-dot"></span>Answers from verified help content</span>
    </div>""")


def welcome() -> None:
    st.html("""
    <div class="nc-hero">
      <h2>How can we help today?</h2>
      <p>Ask about mobile data, billing, SIM and eSIM, roaming, calls or the MyTelecom app,
         or request a refund for a recent recharge.</p>
    </div>""")
    cols = st.columns(2)
    for i, (icon, title, desc, questions) in enumerate(TOPICS):
        with cols[i % 2].container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            for q in questions:
                st.button(q, key=f"topic_{i}_{q}", type="tertiary", icon=":material/arrow_forward:",
                          on_click=ask, args=(q,))
    st.caption("I can't see your account balance or bill. For those, use the MyTelecom app or call 611.")


# ---------- session state ----------
for key, default in {"messages": [], "pending": None, "pending_action": None,
                     "store": None, "assistant": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.store is None:
    st.session_state.store = AccountStore()


def new_assistant() -> Assistant:
    return Assistant(store=st.session_state.store, retriever=load_retriever())


def new_conversation() -> None:  # FR-04, agent FR-48: clears history and any pending refund
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.pending_action = None
    st.session_state.assistant = None


def reset_demo_account() -> None:
    st.session_state.store = AccountStore()
    st.session_state.assistant = None


# ---------- sidebar ----------
with st.sidebar:
    st.html(f"""<div class="nc-side-brand"><div class="nc-logo">{LOGO_SVG}</div>
                <div><div class="nc-name" style="font-size:1rem">{BRAND}</div>
                <div class="nc-tag" style="font-size:.78rem">Customer care assistant</div></div></div>""")
    st.button("New conversation", icon=":material/add_comment:", type="primary", width="stretch",
              on_click=new_conversation)

    st.html('<div class="nc-label">Suggested questions</div>')
    for q in SAMPLE_QUESTIONS:
        st.button(q, key=f"side_{q}", type="tertiary", on_click=ask, args=(q,))

    st.html('<div class="nc-label">Try a refund</div>')
    for q in REFUND_SAMPLES:
        st.button(q, key=f"refund_{q}", type="tertiary", on_click=ask, args=(q,))

    st.html('<div class="nc-label">Your account</div>')
    with st.container(border=True):
        user = st.session_state.store.get_user("user_123")
        st.markdown(f"**{user['name']}** · {user['plan']} plan")
        st.caption("Demo account with simulated data")
        for r in user["recharges"]:
            refunded = r["refund_status"] == "refunded"
            badge = ":green-badge[Refunded]" if refunded else ":gray-badge[Not refunded]"
            days = r["recharge_age_days"]
            st.markdown(f"₹{r['recharge_amount']:,} recharge &nbsp;{badge}  \n"
                        f"<span style='opacity:.6;font-size:.8rem'>{days} day{'s' if days != 1 else ''} ago · "
                        f"{r['recharge_id']}</span>", unsafe_allow_html=True)
        st.button("Reset demo data", icon=":material/restart_alt:", type="tertiary",
                  on_click=reset_demo_account)

    try:
        counts = kb_counts()
    except Exception:  # noqa: BLE001
        counts = {}
        log.exception("Could not open the knowledge base")
    st.html('<div class="nc-label">About</div>')
    if counts:
        st.caption(f"Answers come only from NovaCell's help content: {counts.get('faq', 0)} FAQs, "
                   f"{counts.get('tickets', 0)} resolved cases and {counts.get('guides', 0)} guide sections.")
    else:
        st.caption("Answers come only from NovaCell's FAQ, resolved cases and user guides.")

kb_empty = not any(counts.values())

# ---------- main ----------
header()

if kb_empty:
    try:
        build_knowledge_base()
    except Exception:  # noqa: BLE001
        log.exception("Could not build the knowledge base")
        st.error("Could not build the knowledge base. Run `python ingest_all.py`, then reload.")
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
typed = st.chat_input("Ask a question or request a refund…")
action = st.session_state.pending_action
question = typed or st.session_state.pending
st.session_state.pending = None
st.session_state.pending_action = None
if action:  # Confirm refund / Cancel button
    question = "Yes, proceed" if action == "confirm" else "Cancel"

if not messages and not question:
    welcome()

for i, msg in enumerate(messages):  # FR-03
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            show_buttons = (msg.get("confirm") and i == len(messages) - 1
                            and assistant.agent.awaiting_confirmation and not question)
            render_assistant_body(msg, bool(show_buttons), i)

if question:
    history = list(messages)  # earlier messages, before this one (agent FR-46)
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        start = time.perf_counter()
        sources: list[dict] = []
        trace: list[str] = []
        turn = None
        error = False
        try:
            with st.spinner("Thinking…"):
                turn = assistant.confirm(action == "confirm") if action else assistant.handle(question, history)
            trace = turn.trace
            if turn.route == "rag":
                sources = to_sources(turn.docs)
                answer = st.write_stream(turn.stream)  # FR-05
            else:
                answer = turn.text  # shown as a card after the rerun below
        except MissingApiKeyError as e:
            answer, error = f"⚠️ {e}", True
        except Exception:  # noqa: BLE001 - network / API failures
            # Details go to the server log (private to the app owner), never to visitors.
            log.exception("Error answering a message")
            answer, error = f"⚠️ Sorry, something went wrong reaching the assistant. {FALLBACK_MESSAGE}", True

        latency = time.perf_counter() - start
        answer = answer if isinstance(answer, str) else "".join(answer)
        msg = {"role": "assistant", "content": answer, "sources": sources, "trace": trace,
               "route": turn.route if turn else None, "refund_state": turn.state if turn else None,
               "confirm": bool(turn and turn.needs_confirmation), "error": error}
        if turn and not error:
            msg["id"] = log_answer(question, answer, sources, latency, channel="web", route=turn.route)
        messages.append(msg)
        if error or turn.route == "refund":
            st.rerun()  # redraw: refund cards, confirm buttons and the account panel reflect the new state
        render_sources(sources)
        render_trace(trace)
        render_feedback(msg)
