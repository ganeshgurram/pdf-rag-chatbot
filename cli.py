"""Command-line interface.

    python cli.py ingest docs/a.pdf docs/b.pdf     # add PDFs to the index
    python cli.py ingest docs/                     # every PDF in a folder
    python cli.py chat                             # interactive chat
    python cli.py ask "What is the refund policy?" # single question
    python cli.py search "refund policy"           # retrieval only, no LLM call
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def collect_pdfs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    return out


def print_sources(hits) -> None:
    print("\nSources:")
    for h in hits:
        print(f"  - {h.chunk.source} p.{h.chunk.page}  (score {h.score:.2f})")


def answer(rag, q: str) -> None:
    hits, stream = rag.ask(q)
    for t in stream:
        print(t, end="", flush=True)
    print()
    print_sources(hits)


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF RAG chatbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("paths", nargs="+")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("search").add_argument("query")
    sub.add_parser("chat")
    sub.add_parser("reset")
    args = ap.parse_args()

    from rag import RAGPipeline  # heavy import after arg parsing so --help is fast
    rag = RAGPipeline()

    if args.cmd == "ingest":
        pdfs = collect_pdfs(args.paths)
        if not pdfs:
            sys.exit("No PDF files found.")
        print(rag.ingest([(p.name, p) for p in pdfs]))
    elif args.cmd == "reset":
        rag.reset()
        print("Index cleared.")
    elif args.cmd == "search":
        for h in rag.retrieve(args.query):
            print(f"\n[{h.chunk.source} p.{h.chunk.page}] score={h.score:.3f}\n{h.chunk.text[:400]}")
    elif args.cmd == "ask":
        answer(rag, args.question)
    else:
        print(f"{len(rag.store)} chunks indexed from: {', '.join(rag.store.sources()) or '(none)'}")
        print("Type a question, or 'exit' to quit.\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                print("Bot: ", end="")
                try:
                    answer(rag, q)
                except Exception as e:  # keep the chat alive on API errors
                    print(f"\n[error] {type(e).__name__}: {e}")
                print()


if __name__ == "__main__":
    main()
