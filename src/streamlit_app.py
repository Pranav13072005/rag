"""
app.py

Streamlit web interface for the RAG system.
Run: streamlit run app.py
Deploy: push to Hugging Face Spaces as a Streamlit app.
"""
import os
import subprocess

if not os.path.exists("data/raw"):
    subprocess.run(["python", "src/download_papers.py"])

if not os.path.exists("chroma_db"):
    subprocess.run(["python", "src/ingest.py"])

import streamlit as st
from src.pipeline import RAGPipeline

# ---- Page config ---------------------------------------------------------
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document QA — RAG System")
st.caption(
    "Ask questions about the document corpus. "
    "Answers are grounded in retrieved passages, not LLM memory."
)

# ---- Load pipeline (cached so it only loads once) ------------------------
@st.cache_resource
def load_pipeline():
    return RAGPipeline(use_reranker=True)

with st.spinner("Loading retriever and models..."):
    pipeline = load_pipeline()

st.success("System ready. Ask a question below.")

# ---- Input ---------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Your question:",
        placeholder="e.g. What is the difference between BERT and GPT?"
    )
with col2:
    show_sources = st.checkbox("Show source passages", value=True)

# ---- Query and Display ---------------------------------------------------
if query:
    with st.spinner("Retrieving context and generating answer..."):
        result = pipeline.query(query)

    st.markdown("### 🧠 Answer")
    st.markdown(result["answer"])

    if show_sources:
        st.markdown("### 📎 Retrieved Passages")
        for i, doc in enumerate(result["retrieved_docs"]):
            source = doc.metadata.get("source", "unknown").split("/")[-1]
            page   = doc.metadata.get("page", "?")
            with st.expander(f"Passage {i+1} — {source} (page {page})"):
                st.write(doc.page_content)

    with st.expander("📊 Usage stats"):
        st.json({
            "model": result["model"],
            "n_sources": len(result["retrieved_docs"])
        })