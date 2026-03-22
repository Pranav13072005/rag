"""
src/pipeline.py

The top-level orchestrator. Wires together:
  Retriever -> Generator

This is the object you'll use everywhere: in the UI, in evaluation,
in the benchmark script. One call to pipeline.query() does everything.
"""

# from src.retriever import RAGRetriever
# from src.generator import generate_answer

from retriever import RAGRetriever
from generator import generate_answer
from typing import Dict


class RAGPipeline:
    """
    Encapsulates the full RAG flow.
    query() is the only method you need to call from outside.
    """

    def __init__(self, use_reranker: bool = True):
        """
        use_reranker=True  -> full two-stage pipeline (recommended)
        use_reranker=False -> single-stage, for ablation only
        """
        self.use_reranker = use_reranker
        self.retriever    = RAGRetriever(use_reranker=use_reranker)
        print(f"Pipeline ready. Reranker: {use_reranker}")

    def query(self, question: str) -> Dict:
        """
        Full RAG query:
        1. Retrieve top-k chunks
        2. Generate grounded answer
        3. Return everything (answer + sources + context)

        Returns dict with keys: answer, context, retrieved_docs, model, usage
        """
        # Step 1: retrieve
        docs = self.retriever.retrieve(question)

        # Step 2: generate
        result = generate_answer(question, docs)

        # Step 3: augment result with retrieved docs
        result["retrieved_docs"] = docs
        result["question"]       = question
        return result


if __name__ == "__main__":
    # Quick smoke test
    pipeline = RAGPipeline(use_reranker=True)
    result   = pipeline.query(
        "What problem does the attention mechanism solve in sequence models?"
    )
    print("\n" + "="*60)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for doc in result["retrieved_docs"]:
        print(f"  - {doc.metadata.get('source', '?')} page {doc.metadata.get('page', '?')}")