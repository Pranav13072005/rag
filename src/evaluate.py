"""
src/evaluate.py

RAGAs evaluation harness.

RAGAs is a framework that uses LLMs to evaluate LLM outputs.
It needs a dataset of (question, answer, contexts, ground_truth) tuples.
You build this dataset manually (20-30 QA pairs), run your pipeline
to get answers and contexts, then RAGAs scores everything.
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
from src.pipeline import RAGPipeline
from typing import List, Dict
import json
import os


# ---- Your evaluation dataset -------------------------------------------
# Build 20-30 of these manually. Read your papers, write questions
# whose answers appear directly in the text, and write the correct answer.
# This is the hardest part — takes 30-45 minutes but is essential.

EVAL_QUESTIONS = [
    {
        "question": "What is the core innovation of the Transformer architecture compared to RNNs?",
        "ground_truth": "The Transformer replaces recurrence entirely with self-attention mechanisms, allowing parallel computation of all positions simultaneously rather than sequentially."
    },
    {
        "question": "What does BERT stand for and what training objective does it use?",
        "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers. It uses masked language modeling as its pre-training objective, masking 15% of tokens and predicting them."
    },
    {
        "question": "How many parameters does GPT-3 have?",
        "ground_truth": "GPT-3 has 175 billion parameters."
    },
    # --- ADD 17+ MORE BASED ON YOUR ACTUAL PAPERS ---
    # Pattern: question must be answerable from your PDFs.
    # Ground truth must be a direct paraphrase of what's in the paper.
    # Do NOT add questions whose answers aren't in your corpus.
]


def run_evaluation(
    pipeline: RAGPipeline,
    eval_questions: List[Dict] = None,
    save_path: str = None
) -> Dict:
    """
    Run all eval questions through the pipeline,
    then compute RAGAs metrics.

    Returns the RAGAs score dict.
    """
    if eval_questions is None:
        eval_questions = EVAL_QUESTIONS

    print(f"Running evaluation on {len(eval_questions)} questions...")

    # Collect pipeline outputs for each question
    data = {
        "question":    [],
        "answer":      [],
        "contexts":    [],   # RAGAs needs list of strings, not Document objects
        "ground_truth": []
    }

    for i, item in enumerate(eval_questions):
        print(f"  [{i+1}/{len(eval_questions)}] {item['question'][:60]}...")
        result = pipeline.query(item["question"])

        data["question"].append(item["question"])
        data["answer"].append(result["answer"])
        # RAGAs wants contexts as List[str], not List[Document]
        data["contexts"].append(
            [doc.page_content for doc in result["retrieved_docs"]]
        )
        data["ground_truth"].append(item["ground_truth"])

    # Convert to HuggingFace Dataset (RAGAs requires this format)
    dataset = Dataset.from_dict(data)

    print("\nRunning RAGAs scoring (this calls the LLM multiple times, takes a few minutes)...")
    
    import os
    os.environ["OPENAI_API_KEY"] = "dummy"

    from ragas.llms import llm_factory
    from groq import Groq

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    ragas_llm = llm_factory(
        "llama-3.1-8b-instant",
        client=groq_client
    )
    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ],
        llm=ragas_llm
    )

    result_dict = {
    "faithfulness":      float(sum(scores["faithfulness"]) / len(scores["faithfulness"])),
    "answer_relevancy":  float(sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"])),
    "context_recall":    float(sum(scores["context_recall"]) / len(scores["context_recall"])),
    "context_precision": float(sum(scores["context_precision"]) / len(scores["context_precision"])),
    "n_questions":       len(eval_questions)
}

    if save_path:
        with open(save_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved results to {save_path}")

    return result_dict


def print_results_table(results: Dict):
    """Pretty-print results for README."""
    print("\n" + "="*55)
    print(f"{'Metric':<25} {'Score':>10} {'Target':>10}")
    print("-"*55)
    targets = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
        "context_recall": 0.75,
        "context_precision": 0.70
    }
    for metric, score in results.items():
        if metric == "n_questions":
            continue
        target  = targets.get(metric, "-")
        status  = "✅" if score >= target else "⚠️"
        print(f"{status} {metric:<23} {score:>10.3f} {target:>10.2f}")
    print("="*55)
    print(f"Evaluated on {results['n_questions']} questions")


if __name__ == "__main__":
    pipeline = RAGPipeline(use_reranker=True)
    results  = run_evaluation(pipeline, save_path="results_reranker.json")
    print_results_table(results)