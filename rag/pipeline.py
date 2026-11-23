"""End-to-end orchestration: ingest PDFs -> index -> retrieve -> generate."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .chunker import TokenChunker
from .config import Settings, settings as default_settings
from .embedder import Embedder
from .generator import make_generator
from .loader import load_pdfs
from .retriever import RetrievedChunk, Retriever
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.embedder = Embedder(self.cfg)
        self.chunker = TokenChunker(self.embedder.tokenizer, self.cfg)
        self.retriever = Retriever(self.cfg, self.embedder)
        self.generator = make_generator(self.cfg)
        self.store = VectorStore.load(self.cfg.index_dir) or VectorStore(self.embedder.dim)
        self.history: list[dict] = []

    # ---- ingestion -----------------------------------------------------------
    def ingest(self, items: list[tuple[str, bytes | str | Path]]) -> dict:
        """items: [(file_name, path_or_bytes)]. Re-ingesting a file name replaces it."""
        names = {name for name, _ in items}
        if names & set(self.store.sources()):
            self._drop_sources(names)

        pages = load_pdfs(items)
        chunks = self.chunker.split(pages, start_id=len(self.store))
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.add(chunks, vectors)
            self.store.save(self.cfg.index_dir)
        return {"files": len(items), "pages": len(pages), "chunks": len(chunks), "total_chunks": len(self.store)}

    def _drop_sources(self, names: set[str]) -> None:
        # Copy the surviving vectors out of FAISS instead of re-embedding them.
        old = self.store
        positions = [i for i, c in enumerate(old.chunks) if c.source not in names]
        fresh = VectorStore(self.embedder.dim)
        if positions:
            vectors = np.vstack([old.index.reconstruct(i) for i in positions]).astype("float32")
            keep = [old.chunks[i] for i in positions]
            for new_id, c in enumerate(keep):
                c.id = new_id
            fresh.add(keep, vectors)
        self.store = fresh

    def reset(self) -> None:
        self.store = VectorStore(self.embedder.dim)
        self.store.save(self.cfg.index_dir)
        self.history.clear()

    # ---- querying ------------------------------------------------------------
    def _retrieval_query(self, query: str) -> str:
        # Follow-ups like "what about the second one?" carry little meaning on
        # their own; prepend the previous question so retrieval has context.
        prev = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), None)
        return f"{prev}\n{query}" if prev else query

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.retriever.retrieve(self.store, self._retrieval_query(query))

    def ask(self, query: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Returns (sources, token_stream). History is updated once the stream is consumed."""
        hits = self.retrieve(query)

        def _stream() -> Iterator[str]:
            parts = []
            for t in self.generator.stream_answer(query, hits, self.history):
                parts.append(t)
                yield t
            # Store the plain question (not the excerpts) to keep history small.
            self.history += [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "".join(parts)},
            ]

        return hits, _stream()
