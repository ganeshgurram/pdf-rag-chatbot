# PDF RAG Chatbot: Technical Documentation

This document covers how the chatbot works, which model or algorithm runs at each stage, what parameters it uses and why, how tokenization is done, and which alternatives are better in which situations.

- **Input:** a user **query** (text) and a **knowledge base** of one or more **PDF files**.
- **Output:** a natural-language **answer** grounded in the PDFs, with **citations** (`[file.pdf p.4]`) and a list of the source passages used.

All tunable values live in [rag/config.py](rag/config.py).

---

## 1. Architecture

```
                      ┌──────────────── INDEXING (once per PDF) ────────────────┐
 PDF files ──► [1] Load & clean ──► [2] Tokenize & chunk ──► [3] Embed ──► [4] Store
   (many)        pypdf                bge WordPiece tokenizer    bge-small     FAISS + BM25
                 Page records         384-token chunks           384-d vectors  (index_store/)
                      └─────────────────────────────────────────────────────────┘

                      ┌──────────────── QUERYING (every question) ──────────────┐
 Query ──► [5] Hybrid retrieval ──► [6] Rerank ──► [7] Generate ──► Answer + citations
            dense (FAISS) top-20      cross-encoder     Gemini 3.5 Flash
            sparse (BM25) top-20      MiniLM-L-6        (or Claude), streaming
            fused with RRF            keeps top-6
                      └─────────────────────────────────────────────────────────┘
```

| # | Stage | File | Model / algorithm | Input | Output |
|---|---|---|---|---|---|
| 1 | Load | [loader.py](rag/loader.py) | `pypdf` text extraction | PDF bytes / paths | `Page(source, page, text)` list |
| 2 | Chunk | [chunker.py](rag/chunker.py) | Recursive token splitter (bge WordPiece tokenizer) | Pages | `Chunk(id, source, page, text, n_tokens)` list |
| 3 | Embed | [embedder.py](rag/embedder.py) | `BAAI/bge-small-en-v1.5` | Chunk texts / query | `float32[n, 384]`, L2-normalised |
| 4 | Store | [vectorstore.py](rag/vectorstore.py) | FAISS `IndexFlatIP` + `BM25Okapi` | Chunks + vectors | Index on disk |
| 5 | Retrieve | [retriever.py](rag/retriever.py) | Dense + BM25 + Reciprocal Rank Fusion | Query | ≤ 40 candidate chunk IDs, fused ranking |
| 6 | Rerank | [retriever.py](rag/retriever.py) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | (query, chunk) pairs | Top 6 chunks with relevance scores |
| 7 | Generate | [generator.py](rag/generator.py) | Google `gemini-3.5-flash` (default) or Claude `claude-opus-5-5` | Query + 6 chunks + chat history | Streamed answer with citations |

---

## 2. Stage by stage

### Stage 1: PDF loading and cleaning

| | |
|---|---|
| **Input** | One or more PDFs (uploaded bytes or file paths) |
| **Output** | `list[Page]`: `{source: "policy.pdf", page: 3, text: "..."}` |
| **Tool** | `pypdf.PdfReader().pages[i].extract_text()` |

**What it does:** reads each page's text layer, then cleans it: joins words hyphenated across line breaks (`autho-\nrization` becomes `authorization`), collapses runs of spaces, and keeps paragraph breaks (`\n\n`) because the chunker uses them as natural split points. Each page keeps its file name and page number, so every answer can cite exactly where a fact came from.

**Why pypdf:** pure Python, no system dependencies, installs anywhere (including Windows), and is fast enough for text PDFs.

**Which is better here:**

| Option | Best for | Trade-off |
|---|---|---|
| **pypdf** (used) | Normal digital PDFs | Weak on complex tables and multi-column layouts; can't read scans |
| PyMuPDF (`fitz`) | Faster, better reading order | AGPL licence |
| pdfplumber | Tables, precise layout | Slower |
| Unstructured / Docling | Mixed layouts, tables to structured output | Heavy dependencies |
| OCR (Tesseract, `ocrmypdf`) | **Scanned** PDFs (images only) | Slow; needed only when `extract_text()` returns nothing |

> If a PDF is a scanned image, Stage 1 returns no pages for it. Run it through `ocrmypdf` first.

---

### Stage 2: Tokenization and chunking

| | |
|---|---|
| **Input** | `list[Page]` |
| **Output** | `list[Chunk]`, each ≤ **384 tokens**, with **64 tokens** of overlap |
| **Tokenizer** | The embedding model's own tokenizer (BERT **WordPiece**, uncased, 30,522-token vocabulary) |

#### Why chunk at all?
1. **Embedding models have a hard input limit.** `bge-small-en-v1.5` reads at most **512 tokens**; anything beyond that is **silently truncated** and never embedded.
2. **Precision.** One vector for a whole 30-page PDF averages every topic together. Small chunks give vectors that each mean one thing, so a search lands on the right paragraph.
3. **LLM cost.** Only the 6 most relevant chunks are sent to Claude, not whole documents.

#### Why count *tokens* rather than characters or words?
The limit that matters (512) is in tokens of the embedding model. Character counts are a poor proxy: 1,500 characters of plain English is about 300 tokens, but 1,500 characters of part numbers, URLs or non-English text can exceed 512. **Measuring with the same tokenizer the embedder uses guarantees that no chunk is ever truncated.** The chunker receives `embedder.tokenizer` directly ([pipeline.py](rag/pipeline.py)).

#### How the tokenizer works (WordPiece)
WordPiece splits text into whole words when they are in its vocabulary, and into sub-word pieces (prefixed `##`) when they aren't:

```
"Unauthorized access is prohibited"
→ ["unauthorized", "access", "is", "prohibited"]           4 tokens
"Re-authentication via OAuth2"
→ ["re", "-", "authentication", "via", "o", "##au", "##th", "##2"]   8 tokens
```

The model adds `[CLS]` and `[SEP]` itself, which is why the chunk budget is **384, not 512**: it leaves headroom for special tokens and, more importantly, keeps chunks in the length range the model was mostly trained on (retrieval quality of small embedders drops on very long inputs).

#### Splitting algorithm (recursive, structure-aware)
1. Split each page on **paragraph** breaks.
2. A paragraph longer than 384 tokens is split into **sentences**.
3. A "sentence" still longer than 384 tokens (tables, long lists) is **hard-split on token IDs**.
4. Pieces are **greedily packed** into chunks up to 384 tokens.
5. When a chunk closes, its last pieces (up to **64 tokens**) are **carried into the next chunk** as overlap.
6. Chunks under **20 tokens** (stray headers, page numbers) are dropped.

Chunks never cross page boundaries, so every chunk has exactly one page number to cite.

#### Parameters and why

| Parameter | Value | Why |
|---|---|---|
| `chunk_size_tokens` | 384 | Fits the 512 limit with headroom; about 1–2 paragraphs, enough context to answer a question on its own |
| `chunk_overlap_tokens` | 64 (~17%) | A sentence cut at a boundary still appears whole in one chunk; the 10–20% range is the usual sweet spot (more overlap only adds duplicates) |
| `min_chunk_tokens` | 20 | Removes noise chunks that would match random queries |

#### Which is better

| Strategy | Better when |
|---|---|
| **Recursive token chunking** (used) | General documents; a reliable default |
| Fixed-size token windows | Very uniform text; simplest possible |
| Semantic chunking (split where embedding similarity drops) | Long documents with many topic shifts; costs one embedding per sentence |
| Heading/section-aware chunking | Manuals and contracts with clear numbered sections |
| Smaller chunks (128–256) | Short, factual Q&A ("what is the limit for X?") |
| Larger chunks (512–1024, with a long-context embedder) | Questions that need whole-section reasoning |

---

### Stage 3: Embedding

| | |
|---|---|
| **Input** | Chunk texts (indexing) or the query (search) |
| **Output** | `float32` matrix, shape `(n, 384)`, each row L2-normalised |
| **Model** | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) through `sentence-transformers` |

**Model facts:** BERT-style encoder, about **33M parameters**, **384-dimensional** output, **512-token** max input, CLS pooling. Trained by BAAI with contrastive learning for retrieval. Strong results on the MTEB retrieval benchmark for its size.

**Why this model:**
- **Runs locally on CPU**: no API key, no per-call cost, your documents never leave your machine for indexing.
- **Small and fast**: embeds a few hundred chunks in seconds on a laptop.
- **Built for retrieval** (asymmetric: short query vs. long passage), not just sentence similarity.

**Parameters and why:**

| Parameter | Value | Why |
|---|---|---|
| `normalize_embeddings` | `True` | Unit vectors make inner product equal cosine similarity, so FAISS `IndexFlatIP` gives cosine scores |
| `query_instruction` | `"Represent this sentence for searching relevant passages: "` | bge was trained with this prefix **on queries only**. It moves the query into "search mode" and measurably improves recall. Passages are embedded without it |
| `embed_batch_size` | 32 | Good CPU throughput without large memory spikes |
| `device` | `cpu` | Works everywhere; switch to `cuda` if you have a GPU |

**Which is better:**

| Model | Dim | Better when |
|---|---|---|
| **bge-small-en-v1.5** (used) | 384 | English, CPU, small-to-medium corpora |
| bge-base-en-v1.5 / bge-large-en-v1.5 | 768 / 1024 | Higher accuracy is worth 3–10× slower embedding |
| BAAI/bge-m3 | 1024 | **Multilingual** PDFs or long chunks (8k tokens) |
| intfloat/e5-small-v2, all-MiniLM-L6-v2 | 384 | Similar speed; MiniLM is weaker on retrieval |
| Hosted APIs (Voyage, Cohere, etc.) | 1024+ | Top accuracy, no local compute, but documents are sent to a third party |

> If you change the embedding model, **delete `index_store/` and re-ingest**: vectors from different models are not comparable. Also check its max length and adjust `chunk_size_tokens`.

---

### Stage 4: Vector store and keyword index

| | |
|---|---|
| **Input** | Chunks + their vectors |
| **Output** | `index_store/vectors.faiss` and `index_store/chunks.json` (persisted; BM25 is rebuilt from the chunks on load) |
| **Tools** | FAISS `IndexFlatIP`; `rank_bm25.BM25Okapi` |

**Why two indexes?** They fail in opposite ways:
- **Dense vectors** understand meaning ("terminate employment" matches "dismissal") but are weak on **exact tokens**: clause numbers, product codes, names, acronyms.
- **BM25** matches exact words and rare terms very well but knows nothing about synonyms.

Using both (**hybrid search**) consistently beats either one alone.

**Why `IndexFlatIP`:** exact (brute-force) search, so 100% recall with no training or tuning. For up to about 100k chunks it answers in milliseconds.

**BM25 tokenization:** BM25 uses its own, much simpler tokenizer: lower-cased whole words (`\w+`). Sub-word pieces would hurt here, because BM25's power comes from matching **rare whole words** (IDF weighting), and splitting `oauth2` into `o ##au ##th ##2` would turn a rare, informative term into common fragments.

**Which is better:**

| Store | Better when |
|---|---|
| **FAISS Flat** (used) | ≤ ~100k chunks, single machine |
| FAISS HNSW / IVF-PQ | Millions of chunks; approximate but much faster and smaller |
| Chroma / LanceDB | Want metadata filtering and a DB API with little setup |
| Qdrant / Weaviate / Milvus / pgvector | Multi-user production, built-in hybrid search, horizontal scaling |

---

### Stage 5: Hybrid retrieval with Reciprocal Rank Fusion

| | |
|---|---|
| **Input** | The user query (for follow-ups, the previous question is prepended) |
| **Output** | Up to 40 candidate chunks, ordered by fused score |

**Steps:**
1. Embed the query (with the bge instruction prefix) and take the **top 20** from FAISS.
2. Score all chunks with BM25 and take the **top 20**.
3. Merge with **Reciprocal Rank Fusion**: `score(d) = Σ 1 / (60 + rank_d)` over both lists.

**Why RRF:** cosine scores (0–1) and BM25 scores (0 to ∞) live on different scales and can't be added directly. RRF uses only **ranks**, so no normalisation or weight tuning is needed. A chunk ranked well by both retrievers rises to the top. `k = 60` is the value from the original RRF paper (Cormack et al., 2009) and is robust across datasets.

**Follow-up questions:** "What about the second one?" means little alone, so the previous user question is prepended to the retrieval query. (The LLM still sees the full chat history.)

**Which is better:**

| Approach | Better when |
|---|---|
| **Hybrid + RRF** (used) | General default; handles both meaning and exact terms |
| Dense only | Conversational, paraphrased questions; no exact codes |
| BM25 only | Very technical text full of identifiers; zero ML |
| LLM query rewriting / HyDE / multi-query | Vague or multi-part questions (costs an extra LLM call per question) |

---

### Stage 6: Cross-encoder reranking

| | |
|---|---|
| **Input** | Query + up to 40 candidate chunks |
| **Output** | The **top 6** chunks with a relevance score |
| **Model** | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |

**Why rerank?** The embedder (a **bi-encoder**) encodes the query and each chunk *separately*, which is fast but coarse. A **cross-encoder** reads the query and chunk **together** in one pass (`[CLS] query [SEP] chunk [SEP]`), so attention connects every query word with every chunk word. It is much more accurate but too slow to run over the whole corpus, which is why it only rescores the 40 candidates.

**Model facts:** 6-layer MiniLM, about 22M parameters, trained on MS MARCO passage ranking (real search queries). Uses the same WordPiece tokenizer family. Reranking 40 pairs takes well under a second on CPU.

**Parameters and why:**

| Parameter | Value | Why |
|---|---|---|
| candidates in | ≤ 40 | Enough recall headroom; reranking cost grows linearly |
| `final_top_k` | 6 | 6 × 384 ≈ 2.3k tokens of context: enough evidence for most questions without diluting the prompt with marginal passages |
| `use_reranker` | `True` | Turn off for the fastest possible responses |

**Which is better:**

| Reranker | Better when |
|---|---|
| **ms-marco-MiniLM-L-6-v2** (used) | English, CPU, low latency |
| BAAI/bge-reranker-base / -v2-m3 | Higher accuracy; **multilingual** (v2-m3) |
| Hosted rerank API (Cohere, Voyage) | Best accuracy, no local compute |
| No reranker | Tiny corpora where the top-6 is already obvious |

---

### Stage 7: Answer generation (Gemini by default, Claude optional)

| | |
|---|---|
| **Input** | Query + 6 retrieved chunks (in `<documents>` tags with source and page) + last 6 chat messages |
| **Output** | Streamed answer text with inline citations `[file.pdf p.N]` |
| **Model** | Google **Gemini 3.5 Flash** (`gemini-3.5-flash`) via the official `google-genai` SDK. Switch to Anthropic Claude with `llm_provider = "claude"` |

Both providers get the **same system prompt and the same retrieved excerpts**, so only the model changes. The rest of the pipeline (Stages 1–6) runs locally and is identical either way.

#### Gemini (default)

**Why Gemini 3.5 Flash:** RAG does most of the hard work before the LLM is called. Retrieval and reranking already hand over the 6 right passages, so the model's job is mostly to **read, combine and cite**. A fast Flash-tier model does that well at low cost and latency. In testing it answered every question correctly with citations, and said "the documents do not cover it" for an out-of-scope question instead of inventing an answer.

**Parameters and why:**

| Parameter | Value | Why |
|---|---|---|
| `gemini_model` | `gemini-3.5-flash` | Fast and reliable. In testing, `gemini-3.8-flash` was often overloaded (503) or stalled |
| `gemini_fallback_models` | `gemini-3.8-flash`, `gemini-2.5-flash` | If the main model is overloaded (503), rate-limited (429) or stalls, the next model is tried automatically, but only if no text was streamed yet, so the user never sees two half-answers |
| `system_instruction` | grounding rules (below) | Gemini's equivalent of a system prompt |
| `max_output_tokens` | 16000 | A ceiling, not a target, so long answers are never cut off; you pay only for tokens generated |
| `temperature` | not set (model default) | Google recommends the default for Gemini 3-series models; grounding comes from the prompt and the excerpts, not low temperature |
| retry options | 2 attempts, 1–4 s back-off, 45 s read timeout | The SDK default (5 attempts, up to 60 s apart) could stall for minutes on an overloaded model; short retries plus model fallback recover in seconds |
| streaming | `client.models.generate_content_stream(...)` | Text appears token by token in the UI |

**Chat history format:** stored provider-neutrally as `user` / `assistant` and converted to Gemini's `user` / `model` roles on each request.

#### Claude (optional)

Set `llm_provider = "claude"` in [rag/config.py](rag/config.py) and put `ANTHROPIC_API_KEY` in `.env`.

| Parameter | Value | Why |
|---|---|---|
| `claude_model` | `claude-opus-5-5` | Most reliable at grounded, cited answers when faithfulness matters most |
| `max_tokens` | 16000 | Same ceiling as above |
| `thinking` | `{"type": "adaptive"}` | The model decides how much to reason: little for lookups, more when reconciling several passages |
| `output_config.effort` | `"medium"` | Balanced cost and latency; `"high"` for complex cross-document analysis, `"low"` for simple lookups |
| `fallbacks` | `"default"` (beta `server-side-fallback-2026-07-01`) | If a safety classifier declines, the API retries on a fallback model in the same call |
| `temperature` | not set | Not accepted by current Claude models; `effort` is the control |

**Prompt design** ([generator.py](rag/generator.py)):
- **System prompt** sets the rules: answer only from the excerpts, cite `[file p.N]`, say clearly when the documents don't cover the question, flag disagreements between documents, and treat document text as data rather than instructions (a defence against prompt injection hidden inside PDFs).
- **Excerpts** go in XML-style `<document index source page>` tags, which both models parse reliably and which carry the citation metadata next to each passage.
- **History** stores only the plain previous questions and answers (not the old excerpts), which keeps follow-up turns small and cheap.

**LLM tokenizers:** Gemini and Claude each use their own server-side sub-word tokenizer, unrelated to the WordPiece tokenizer above. It only matters for **cost and context limits**, not for retrieval. Rough guide for English: 1 token ≈ 4 characters. A typical request here is about 2.5k–4k input tokens (6 chunks + prompt + history). For exact counts use `client.models.count_tokens(...)` (Gemini) or `client.messages.count_tokens(...)` (Claude).

**Which is better:**

| Model | Better when |
|---|---|
| **Gemini 3.5 Flash** (default) | Everyday document Q&A; fast and low cost |
| Gemini 3.8 Flash | Newer Flash model; use as primary when it isn't overloaded |
| Gemini Pro (`gemini-pro-latest`, `gemini-3.1-pro-preview`) | Harder reasoning across many documents; slower and pricier, and may need a paid tier |
| Gemini Flash-Lite (`gemini-3.5-flash-lite`) | Very high volume, simple lookups, lowest cost |
| Claude Opus 5.5 (`claude-opus-5-5`) | Faithfulness matters most (legal, compliance, policy answers) |
| Claude Sonnet 5.5 / Haiku 5.5 | Claude at lower cost for high volume |

To switch, change `gemini_model` (or `llm_provider` / `claude_model`) in [rag/config.py](rag/config.py). Run `client.models.list()` to see which Gemini models your key can use.

---

## 3. Tokenization summary

Three different tokenizers are used, each chosen for its job:

| Where | Tokenizer | Unit | Why this one |
|---|---|---|---|
| Chunking + embedding (Stages 2–3) | bge **WordPiece** (BERT uncased, 30,522 vocab) | Sub-words | Must match the embedder exactly, so chunk sizes are measured in the units the 512-token limit is defined in |
| Reranking (Stage 6) | MiniLM WordPiece (same family) | Sub-words | Built into the cross-encoder; 384-token chunks + a short query stay under its 512 limit |
| BM25 keyword search (Stages 4–5) | Regex `\w+`, lower-cased | Whole words | BM25 relies on rare whole words (IDF); sub-word pieces would destroy that signal |
| Generation (Stage 7) | Gemini's (or Claude's) own tokenizer, server-side | Sub-words | Fixed by the model; only affects cost and context size |

---

## 4. Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then put your GEMINI_API_KEY in .env
```

**Web chat UI** (upload several PDFs in the sidebar, click *Index*, then chat):
```bash
streamlit run app.py
```

**Command line:**
```bash
python cli.py ingest docs/                   # index every PDF in a folder
python cli.py chat                           # interactive chat
python cli.py ask "What does section 4 say about access control?"
python cli.py search "access control"        # retrieval only, no API call
python cli.py reset                          # clear the index
```

The first run downloads the two local models (about 130 MB and 90 MB) from Hugging Face; after that they are cached.

---

## 5. Tuning guide

| Symptom | Change |
|---|---|
| "Documents don't cover it" but they do | Raise `dense_top_k`/`sparse_top_k` to 40 and `final_top_k` to 8–10; try smaller chunks (256) |
| Answers miss surrounding context | Larger chunks (512 with a long-context embedder) or `final_top_k` 8–10 |
| Exact IDs or clause numbers not found | Hybrid search already helps; also check Stage 1 extracted them (`python cli.py search "<id>"`) |
| Non-English PDFs | `embedding_model = "BAAI/bge-m3"`, `reranker_model = "BAAI/bge-reranker-v2-m3"`, clear the query instruction, re-ingest |
| Too slow | `use_reranker = False`; if you see `falling back to ...` messages often, put a less busy model first in `gemini_model` |
| `503 UNAVAILABLE` / `429` errors | The model is overloaded or you hit a free-tier limit; fallback handles it automatically, or wait and retry |
| Too expensive | A Flash-Lite model; reduce `final_top_k` |
| Answers not faithful enough | A Gemini Pro model, or `llm_provider = "claude"` |
| Scanned PDFs return nothing | OCR them first with `ocrmypdf input.pdf output.pdf` |
