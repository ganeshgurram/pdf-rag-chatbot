# PDF RAG Chatbot

Ask questions about multiple PDF files and get answers grounded in them, with page-level citations.

**Pipeline:** pypdf → token-aware chunking → `bge-small-en-v1.5` embeddings → FAISS + BM25 hybrid search → cross-encoder rerank → Gemini 3.5 Flash (or Claude, set in `rag/config.py`).

```bash
pip install -r requirements.txt
copy .env.example .env      # add GEMINI_API_KEY
streamlit run app.py        # web UI
python cli.py ingest docs/  # or the CLI
python cli.py chat
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for the models, parameters, tokenization, and per-stage trade-offs.
