from src.pipeline import RAGPipeline
from typing import List, Dict
import numpy as np


# ----------- DEFINE DOMAIN LABELS FOR PAPERS -----------
NLP_PAPERS = [
    "attention_is_all_you_need",
    "bert",
    "roberta",
    "albert",
    "t5",
    "xlnet",
    "gpt3",
    "llama",
    "longformer",
    "reformer",
    "performer",
    "lora",
    "chinchilla",
    "vision_transformer"
]

RL_PAPERS = [
    "dqn",
    "ddpg",
    "ppo",
    "sac",
    "rainbow",
    "decision_transformer"
]


# ----------- SIMPLE TOKEN OVERLAP METRIC -----------
def token_overlap(a: str, b: str):
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    return len(a_tokens & b_tokens) / max(len(a_tokens), 1)


# ----------- MAIN EVALUATION FUNCTION -----------
def evaluate_pipeline(pipeline: RAGPipeline, eval_data: List[Dict]):

    domain_correct = []
    groundedness = []

    for item in eval_data:
        q = item["question"]
        gt = item["ground_truth"]
        domain = item["domain"]

        result = pipeline.query(q)
        docs = result["retrieved_docs"]
        answer = result["answer"]

        # --- Domain accuracy ---
        retrieved_sources = [
            doc.metadata.get("source", "").lower()
            for doc in docs
        ]

        if domain == "nlp":
            correct = any(p in s for s in retrieved_sources for p in NLP_PAPERS)
        else:
            correct = any(p in s for s in retrieved_sources for p in RL_PAPERS)

        domain_correct.append(int(correct))

        # --- Groundedness ---
        context_text = " ".join([d.page_content for d in docs])
        groundedness.append(token_overlap(answer, context_text))

    return {
        "domain_accuracy": np.mean(domain_correct),
        "groundedness": np.mean(groundedness)
    }


if __name__ == "__main__":

    eval_data = [

# ---------- NLP (10) ----------

{"question": "What replaces recurrence in the Transformer?",
 "ground_truth": "Self-attention replaces recurrence.",
 "domain": "nlp"},

{"question": "What is masked language modeling in BERT?",
 "ground_truth": "It predicts masked tokens during pretraining.",
 "domain": "nlp"},

{"question": "What is the key idea of RoBERTa?",
 "ground_truth": "Improved BERT training with more data and dynamic masking.",
 "domain": "nlp"},

{"question": "What does ALBERT reduce?",
 "ground_truth": "Parameter count using factorized embeddings.",
 "domain": "nlp"},

{"question": "What is T5 framing of NLP tasks?",
 "ground_truth": "All tasks are converted to text-to-text format.",
 "domain": "nlp"},

{"question": "What is XLNet training objective?",
 "ground_truth": "Permutation language modeling.",
 "domain": "nlp"},

{"question": "Why is Longformer efficient?",
 "ground_truth": "It uses sparse attention for long sequences.",
 "domain": "nlp"},

{"question": "What does LoRA modify?",
 "ground_truth": "It adds low-rank adapters to transformer weights.",
 "domain": "nlp"},

{"question": "What is scaling law insight from Chinchilla?",
 "ground_truth": "Optimal performance requires balanced data and parameters.",
 "domain": "nlp"},

{"question": "What does Vision Transformer treat image patches as?",
 "ground_truth": "Sequence tokens processed by transformer.",
 "domain": "nlp"},

# ---------- RL (10) ----------

{"question": "What is Deep Q-Network?",
 "ground_truth": "It combines Q-learning with deep neural networks.",
 "domain": "rl"},

{"question": "What does DDPG learn?",
 "ground_truth": "Deterministic policy using actor-critic method.",
 "domain": "rl"},

{"question": "What is PPO objective?",
 "ground_truth": "Clipped surrogate objective stabilizes policy updates.",
 "domain": "rl"},

{"question": "What is Soft Actor-Critic entropy term used for?",
 "ground_truth": "Encourages exploration during training.",
 "domain": "rl"},

{"question": "What does Rainbow combine?",
 "ground_truth": "Multiple DQN improvements into one algorithm.",
 "domain": "rl"},

{"question": "What is Decision Transformer idea?",
 "ground_truth": "Model reinforcement learning as sequence prediction.",
 "domain": "rl"},

{"question": "Why use replay buffer in DQN?",
 "ground_truth": "To decorrelate training samples.",
 "domain": "rl"},

{"question": "What is policy gradient method?",
 "ground_truth": "Optimizes policy directly via gradient ascent.",
 "domain": "rl"},

{"question": "What is value function in RL?",
 "ground_truth": "Expected return from a state.",
 "domain": "rl"},

{"question": "Why is off-policy learning useful?",
 "ground_truth": "Allows reuse of past experience.",
 "domain": "rl"},

# ---------- MIXED (5) ----------

{"question": "How are transformers used in reinforcement learning?",
 "ground_truth": "Transformers model trajectories as sequences.",
 "domain": "rl"},

{"question": "Why sequence modeling useful in decision making?",
 "ground_truth": "Actions depend on past state sequences.",
 "domain": "rl"},

{"question": "What connects attention and RL planning?",
 "ground_truth": "Attention helps model long-term dependencies.",
 "domain": "nlp"},

{"question": "Why transformer useful for control tasks?",
 "ground_truth": "They model temporal dependencies efficiently.",
 "domain": "rl"},

{"question": "What is advantage of parallel computation in transformers?",
 "ground_truth": "Allows faster sequence processing.",
 "domain": "nlp"},
]

    pipe_a = RAGPipeline(use_reranker=True)
    scores_a = evaluate_pipeline(pipe_a, eval_data)

    pipe_b = RAGPipeline(use_reranker=False)
    scores_b = evaluate_pipeline(pipe_b, eval_data)

    print("\nWITH RERANKER:", scores_a)
    print("WITHOUT RERANKER:", scores_b)

    print("\nRERANKER GAIN")
    print({
        "domain_gain": scores_a["domain_accuracy"] - scores_b["domain_accuracy"],
        "groundedness_gain": scores_a["groundedness"] - scores_b["groundedness"]
    })