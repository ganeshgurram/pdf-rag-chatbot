"""Central configuration. Every tunable used by the pipeline lives here so the
documentation (DOCUMENTATION.md) can point at one place."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # ---- Storage -----------------------------------------------------------
    index_dir: Path = Path("index_store")

    # ---- Stage 2: chunking (token-based, measured with the EMBEDDING model's tokenizer)
    chunk_size_tokens: int = 384      # < 512 = bge-small max sequence length (leaves room for special tokens)
    chunk_overlap_tokens: int = 64    # ~17% overlap so sentences cut at a boundary appear in both chunks
    min_chunk_tokens: int = 20        # drop near-empty fragments (page numbers, headers)

    # ---- Stage 3: embeddings ----------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"   # 384-dim, 33M params, CPU friendly
    # bge models are trained with this instruction prefix on the QUERY side only.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 32

    # ---- Stage 5: retrieval -----------------------------------------------
    dense_top_k: int = 20             # candidates from the vector index
    sparse_top_k: int = 20            # candidates from BM25 keyword search
    rrf_k: int = 60                   # Reciprocal Rank Fusion constant (standard value from the RRF paper)

    # ---- Stage 6: reranking -----------------------------------------------
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    final_top_k: int = 6              # chunks actually sent to the LLM

    # ---- Stage 7: generation ----------------------------------------------
    llm_provider: str = "gemini"      # "gemini" (GEMINI_API_KEY) or "claude" (ANTHROPIC_API_KEY)
    gemini_model: str = "gemini-3.5-flash"
    # Tried in order when the main model is overloaded (503), rate-limited (429) or stalls.
    gemini_fallback_models: tuple[str, ...] = ("gemini-3.8-flash", "gemini-2.5-flash")
    claude_model: str = "claude-opus-5-5"
    max_tokens: int = 16000           # ceiling on answer length, not a target
    effort: str = "medium"            # Claude only: low | medium | high | xhigh | max
    history_turns: int = 6            # previous chat messages kept for follow-up questions

    extra: dict = field(default_factory=dict)


settings = Settings()
