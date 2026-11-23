"""Stage 2 - Tokenization + chunking.

Input : list[Page]
Output: list[Chunk] - overlapping windows of at most `chunk_size_tokens` tokens

Chunk length is measured with the *embedding model's own tokenizer* (WordPiece
for bge). That guarantees no chunk is silently truncated by the embedder, which
would otherwise throw away the tail of the text without any error.

Splitting is recursive: paragraphs first, then sentences, then words, so a chunk
boundary falls on the most natural break that still fits the budget.
"""

import re
from dataclasses import asdict, dataclass

from transformers import PreTrainedTokenizerBase

from .config import Settings
from .loader import Page

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: int
    source: str
    page: int
    text: str
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


class TokenChunker:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: Settings):
        self.tok = tokenizer
        self.size = cfg.chunk_size_tokens
        self.overlap = cfg.chunk_overlap_tokens
        self.min_tokens = cfg.min_chunk_tokens

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _units(self, text: str) -> list[str]:
        """Break text into pieces that each fit inside one chunk."""
        out: list[str] = []
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            if self.count(para) <= self.size:
                out.append(para)
                continue
            for sent in _SENTENCE_RE.split(para):
                if self.count(sent) <= self.size:
                    out.append(sent)
                else:  # a single giant "sentence" (tables, lists) -> hard split on tokens
                    ids = self.tok.encode(sent, add_special_tokens=False)
                    step = self.size - self.overlap
                    for i in range(0, len(ids), step):
                        out.append(self.tok.decode(ids[i : i + self.size]))
        return out

    def _pack(self, units: list[str]) -> list[str]:
        """Greedily pack units into chunks, carrying `overlap` tokens forward."""
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for u in units:
            n = self.count(u)
            if cur and cur_len + n > self.size:
                chunks.append(" ".join(cur))
                # carry the tail of the previous chunk as overlap
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(cur):
                    pl = self.count(prev)
                    if tail_len + pl > self.overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += pl
                cur, cur_len = tail, tail_len
            cur.append(u)
            cur_len += n
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def split(self, pages: list[Page], start_id: int = 0) -> list[Chunk]:
        chunks: list[Chunk] = []
        next_id = start_id
        for p in pages:
            for text in self._pack(self._units(p.text)):
                n = self.count(text)
                if n < self.min_tokens:
                    continue
                chunks.append(Chunk(id=next_id, source=p.source, page=p.page, text=text, n_tokens=n))
                next_id += 1
        return chunks
