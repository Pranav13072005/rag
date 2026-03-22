"""
src/ingest.py

Loads all PDFs from data/raw/, splits them into overlapping chunks,
embeds each chunk using a HuggingFace model, and stores the
vectorstore on disk using ChromaDB.

This file is run ONCE before training. Never needs to run again
unless you add new documents.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from tqdm import tqdm

# ---- Configuration -------------------------------------------------------
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"   # 384-dim, fast, excellent quality
CHROMA_DIR   = "./chroma_db"              # where vectorstore lives on disk
DATA_DIR     = "./data/raw"               # where your PDFs live
CHUNK_SIZE   = 512                        # tokens per chunk
CHUNK_OVERLAP = 50                        # overlap between adjacent chunks
# --------------------------------------------------------------------------


def build_vectorstore(chunk_size: int = CHUNK_SIZE,
                      chunk_overlap: int = CHUNK_OVERLAP) -> Chroma:
    """
    Full ingestion pipeline:
    1. Load PDFs
    2. Split into chunks
    3. Embed chunks
    4. Store in ChromaDB
    """

    # --- Step 1: Load PDFs ------------------------------------------------
    print("[1/4] Loading PDFs...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    print(f"      Loaded {len(docs)} pages from {DATA_DIR}")

    if len(docs) == 0:
        raise ValueError(f"No PDFs found in {DATA_DIR}. Add PDFs first.")

    # --- Step 2: Split into chunks ----------------------------------------
    print(f"[2/4] Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,              # count characters, not tokens
        separators=["\n\n", "\n", ". ", " ", ""]  # split on paragraphs first
    )
    chunks = splitter.split_documents(docs)
    print(f"      Created {len(chunks)} chunks")

    # --- Step 3: Load embedding model ------------------------------------
    print(f"[3/4] Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},    # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}  # L2 normalise for cosine sim
    )

    # --- Step 4: Build and persist ChromaDB -------------------------------
    print(f"[4/4] Building vectorstore at {CHROMA_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="rag_collection"
    )

    vectorstore.persist()
    print(f"      Done. {len(chunks)} chunks stored in {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()