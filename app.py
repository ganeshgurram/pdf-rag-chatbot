"""Streamlit chat UI.  Run:  streamlit run app.py"""

from dotenv import load_dotenv
import streamlit as st

from rag import RAGPipeline

load_dotenv()
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding + reranker models...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


rag = get_pipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]

# ---- Sidebar: knowledge base -------------------------------------------------
with st.sidebar:
    st.header("Knowledge base")
    files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index uploaded PDFs", type="primary", disabled=not files):
        with st.spinner("Extracting, chunking and embedding..."):
            try:
                stats = rag.ingest([(f.name, f.getvalue()) for f in files])
                st.success(f"Indexed {stats['files']} file(s): {stats['pages']} pages -> {stats['chunks']} chunks")
            except ValueError as e:
                st.error(str(e))

    st.caption(f"{len(rag.store)} chunks indexed")
    for s in rag.store.sources():
        st.write(f"- {s}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        rag.history.clear()
        st.rerun()
    if st.button("Delete index"):
        rag.reset()
        st.session_state.messages = []
        st.rerun()

# ---- Chat ---------------------------------------------------------------------
st.title("📄 Chat with your PDFs")


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']} p.{s['page']}** · rerank score {s['score']:.2f}")
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if query := st.chat_input("Ask a question about your documents"):
    if len(rag.store) == 0:
        st.warning("Upload and index at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            hits, stream = rag.ask(query)
            answer = st.write_stream(stream)
        except Exception as e:  # surface API/auth errors in the UI
            st.error(f"{type(e).__name__}: {e}")
            st.stop()
        sources = [
            {"source": h.chunk.source, "page": h.chunk.page, "score": h.score, "text": h.chunk.text}
            for h in hits
        ]
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
