# Telecom Support Assistant (RAG chatbot)

A chatbot that answers Tier-1 telecom support questions using only three knowledge sources:
the FAQ, resolved support tickets and the PDF user guide. It is built to the spec in [PRD.md](PRD.md).

- **Retrieval:** ChromaDB, top-3 results from each of `faq`, `tickets` and `guides` (9 documents per question), embedded locally with `all-MiniLM-L6-v2`
- **Generation:** `qwen/qwen3.8-27b` on Groq (the PRD names `qwen3.6-27b`, which Groq no longer serves; set `GROQ_MODEL` in `.env` to change it) with `temperature=0` and `reasoning_effort="none"`, using a LangChain LCEL chain
- **Interfaces:** a Streamlit web UI (`app.py`) and a CLI (`main.py`)
- **Refund agent:** intent detection routes refund requests to a tool-using Refund Agent with deterministic validation and a confirmation step. See [agent_PRD.md](agent_PRD.md) and the section below.

## Setup (Windows)

Everything installs into a project-local `.venv`.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
copy .env.example .env        # then put your key in GROQ_API_KEY=
.venv\Scripts\python ingest_all.py
```

Get a free Groq key at https://console.groq.com. The embedding model (about 90 MB) downloads automatically on the first ingest.

## Run

```bash
.venv\Scripts\streamlit run app.py
```

```bash
.venv\Scripts\python main.py
```

## Updating knowledge (support ops)

| Change | Then run |
|---|---|
| Edit `data/faq.csv` | `.venv\Scripts\python ingest_faq.py` |
| Add resolved tickets to `data/tickets.db` | `.venv\Scripts\python ingest_tickets.py` |
| Replace `data/telecom_guide.pdf` | `.venv\Scripts\python ingest_guides.py` |

Ingest scripts can safely be re-run: each one rebuilds its collection in `chroma_store/`. Only tickets with `status = 'resolved'` are indexed. The app does not re-ingest when it starts.

To add a new source, write an `ingest_<name>.py` that fills a Chroma collection, then add the collection to `COLLECTIONS` in [retriever.py](retriever.py).

## Refund agent

Every message first goes through intent detection (`intent.py`, using the same Groq model):

- `requirement_inquiry` goes to the existing RAG pipeline, unchanged. The last 5 exchanges are passed to the LLM as context, but retrieval still uses only the latest message.
- `refund_request` goes to the Refund Agent (`refund_agent.py`).

The LLM's role in the agent is limited to calling `get_user_account` and `validate_refund` to work out which amount the customer means. Everything that affects money is done in code (`refund_tools.py`):

| Rule | Where it is enforced |
|---|---|
| Recharge must be completed, at most 7 days old, not already refunded; amount > 0 and ≤ recharge | `validate_refund` |
| Which recharge a refund applies to (an exact amount match wins, even if already refunded) | `select_recharge` |
| ≤ ₹499: refunded only after an explicit "Yes, proceed" or the **Confirm Refund** button | `RefundAgent.resolve_confirmation`, the only caller of `process_refund` |
| > ₹499: submitted to the Support Team; never refunded here | `submit_to_support` (refund status unchanged) |
| The model can't invent an amount: it must appear in the customer's own words | amount guard in `RefundAgent._plan_and_decide` |

`process_refund` also re-checks validation and the ₹499 limit itself, so it can never refund an invalid amount even if it were called by mistake. The refund is simulated: it updates the in-memory demo account and returns a `ref_…` reference.

The demo user is `user_123` (Rahul) with recharges of ₹999 (2 days old) and ₹499 (1 day old). It is hard-coded in `refund_data.py` and treated as the signed-in session user. In the web UI, **Demo account (simulated)** in the sidebar shows refund status and has a reset button. **Clear conversation** resets the chat history and any pending refund.

Workflow steps (`[Intent] refund_request`, `[Agent] validate_refund(amount=499)` and so on) are shown under each answer in **Workflow steps** and written to `logs/agent.log`.

Tests:

```bash
.venv\Scripts\python -m unittest test_refund_agent -v
```

```bash
.venv\Scripts\python demo_scenarios.py
```

The first runs deterministic tests with no LLM or network. The second runs the PRD's 7 demonstration scenarios against the live model.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, model names, embeddings, Chroma helpers |
| `ingest_*.py` | Load FAQ / tickets / guide into `chroma_store/` |
| `retriever.py` | Parallel merged retriever and source-labelled context |
| `chain.py` | Grounded prompt, Groq LLM, streaming, conversation history |
| `assistant.py` | Routes each message to RAG or the Refund Agent |
| `intent.py` | `requirement_inquiry` / `refund_request` classifier |
| `refund_agent.py` | Refund Agent: tool calling, state machine, confirmation |
| `refund_tools.py` | Deterministic tools: account lookup, validation, simulated refund, support submission |
| `refund_data.py` | Hard-coded demo account |
| `workflow_log.py` | Step logging to `logs/agent.log` |
| `app.py` | Streamlit UI: sample questions, refund samples, confirm buttons, sources, workflow steps, 👍/👎 |
| `main.py` | CLI REPL (`quit` to exit, `clear` to reset; reply "Yes, proceed" or "Cancel" to confirm) |
| `interaction_log.py` | Writes answers and feedback to `logs/interactions.jsonl` |
| `eval_samples.py` | Runs the sample questions and reports the share answered without a "call 611" fallback |
| `test_refund_agent.py` | Deterministic refund tests |
| `demo_scenarios.py` | Live run of the agent PRD's demonstration scenarios |
