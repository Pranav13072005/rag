"""
notebooks/ablation.py

Runs the evaluation twice:
  Run A: with reranker (two-stage)
  Run B: without reranker (single-stage)

Prints a comparison table showing the delta.
This is your main experiment result.
"""

import json
from src.pipeline import RAGPipeline
from src.evaluate import run_evaluation, print_results_table


def run_ablation():
    print("=" * 60)
    print("ABLATION STUDY: With Reranker vs Without Reranker")
    print("=" * 60)

    # Run A: full pipeline with reranker
    print("\n[Run A] Two-stage pipeline (bi-encoder + reranker)")
    pipeline_a = RAGPipeline(use_reranker=True)
    results_a  = run_evaluation(pipeline_a, save_path="results_with_reranker.json")
    print("\nResults A (WITH reranker):")
    print_results_table(results_a)

    # Run B: single-stage without reranker
    print("\n[Run B] Single-stage pipeline (bi-encoder only)")
    pipeline_b = RAGPipeline(use_reranker=False)
    results_b  = run_evaluation(pipeline_b, save_path="results_no_reranker.json")
    print("\nResults B (WITHOUT reranker):")
    print_results_table(results_b)

    # Print delta table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} {'With Reranker':>14} {'No Reranker':>12} {'Delta':>8}")
    print("-" * 65)
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    for m in metrics:
        a, b  = results_a[m], results_b[m]
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        print(f"{m:<25} {a:>14.3f} {b:>12.3f} {sign}{delta:>7.3f}")
    print("=" * 65)
    print("\nSave this table. It goes directly in your README.")


if __name__ == "__main__":
    run_ablation()