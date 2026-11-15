# Telecom Support Assistant (RAG chatbot)

A chatbot that answers Tier-1 telecom support questions using only three knowledge sources:
the FAQ, resolved support tickets and the PDF user guide. It is built to the spec in [PRD.md](PRD.md).

- **Retrieval:** ChromaDB, top-3 results from each of `faq`, `tickets` and `guides` (9 documents per question), embedded locally with `all-MiniLM-L6-v2`
- **Generation:** `qwen/qwen3.8-27b` on Groq (the PRD names `qwen3.6-27b`, which Groq no longer serves; set `GROQ_MODEL` in `.env` to change it) with `temperature=0` and `reasoning_effort="none"`, using a LangChain LCEL chain
- **Interfaces:** a Streamlit web UI (`app.py`) and a CLI (`main.py`)

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

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, model names, embeddings, Chroma helpers |
| `ingest_*.py` | Load FAQ / tickets / guide into `chroma_store/` |
| `retriever.py` | Parallel merged retriever and source-labelled context |
| `chain.py` | Grounded prompt, Groq LLM, streaming |
| `app.py` | Streamlit UI: sample questions, clear button, sources expander, 👍/👎 feedback |
| `main.py` | CLI REPL (type `quit` to exit) |
| `interaction_log.py` | Writes answers and feedback to `logs/interactions.jsonl` |
| `eval_samples.py` | Runs the sample questions and reports the share answered without a "call 611" fallback |
