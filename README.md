# Domain-Adaptive Retrieval-Augmented Generation with Cross-Encoder Reranking: A Multi-Domain Evaluation Study

> **Live Demo:** https://huggingface.co/spaces/pranav070/rag
> 

> **Corpus:** 20 research papers across NLP, Transformers, and Reinforcement Learning
> 

> **Evaluation:** 25 cross-domain questions | Domain Accuracy + Groundedness metrics
> 

---

## Abstract

We build and evaluate a two-stage Retrieval-Augmented Generation (RAG) pipeline over a heterogeneous multi-domain corpus of 20 research papers spanning Natural Language Processing, Transformer architectures, and Reinforcement Learning. The system combines bi-encoder dense retrieval (BAAI/bge-small-en-v1.5) with optional cross-encoder reranking (ms-marco-MiniLM-L-6-v2) and Llama-3-8B generation via the Groq API. We evaluate across two axes: **domain accuracy** (whether retrieved context belongs to the correct sub-domain) and **groundedness** (whether the generated answer is faithful to retrieved evidence).

Our results reveal a previously under-studied **precision-groundedness tradeoff**: reranking improves answer groundedness (+2.09 points) while reducing domain-level retrieval accuracy (-4.0 points), suggesting that cross-encoder reranking optimises for local lexical relevance at the cost of domain-level semantic coherence in heterogeneous corpora.

---

## Results

| Configuration | Domain Accuracy (↑) | Groundedness (↑) |
| --- | --- | --- |
| Single-stage (bi-encoder only) | **0.880** | 0.4997 |
| Two-stage (+ cross-encoder reranker) | 0.840 | **0.5207** |
| **Delta (reranker effect)** | **-0.040** | **+0.0209** |

### What These Numbers Mean

**Groundedness improves with reranking (+2.09%).** The cross-encoder selects passages with stronger query-passage alignment, giving the generator more precisely targeted evidence. The LLM stays closer to what it was given, reducing reliance on parametric (memorised) knowledge.

**Domain accuracy drops with reranking (-4.0%).** This is the more interesting finding. The cross-encoder (trained on MS MARCO web search data) has a bias toward passages with high surface-form lexical overlap with the query — regardless of domain. In a heterogeneous corpus, queries about "policy gradient optimisation in RL" share vocabulary with NLP passages about "gradient-based learning" and "loss optimisation". The cross-encoder promotes these cross-domain false positives above genuinely relevant RL passages.

The bi-encoder, having embedded each passage into a semantic space during ingestion, preserves more domain-level geometric structure. Its coarser but domain-aware similarity metric is less susceptible to cross-domain lexical traps.

**Takeaway:** This is not a system failure — it is a **precision-groundedness tradeoff** that exists in multi-domain corpora and is not commonly discussed in RAG literature. The right configuration depends on what you are optimising for.

---

## System Architecture

```
OFFLINE (once)
20 PDFs (NLP×7, Transformers×7, RL×6)
    → PyPDFLoader
    → RecursiveCharacterTextSplitter (chunk=512, overlap=50)
    → HuggingFaceEmbeddings (bge-small-en-v1.5, 384-dim)
    → ChromaDB (persisted to disk)

ONLINE (per query)
Query → bge-small embed → ChromaDB ANN search (top-20 candidates)
    [Optional] → CrossEncoder (ms-marco-MiniLM-L-6-v2) → reranked top-4
    → Groq API (llama3-8b-8192, temp=0.1) + grounding system prompt
    → Grounded answer with source citations
```

---

## Corpus

| Domain | Papers (n) | Representative Topics |
| --- | --- | --- |
| Natural Language Processing | 7 | Seq2seq, BERT, ELMo, word embeddings, text classification |
| Transformer Architectures | 7 | Attention, GPT, Vision Transformer, T5, BART |
| Reinforcement Learning | 6 | PPO, RLHF, DQN, policy gradient, Q-learning |
| **Total** | **20** | — |

Corpus is deliberately imbalanced (7:7:6) to study minority-domain retrieval behaviour. RL is the smallest domain and shows the most cross-domain contamination from the reranker.

---

## Evaluation Design

**25 questions manually curated** — 9 NLP, 9 Transformer, 7 RL. Questions are designed to require retrieval: they reference specific paper results, architectural details, and numerical claims that cannot be answered from general LLM knowledge alone.

**Domain Accuracy:** Whether retrieved top-k passages belong to the same domain as the question. Measures cross-domain contamination in retrieval.

**Groundedness:** Whether every claim in the generated answer is entailed by the retrieved context. Scored using an LLM judge (0–1 scale). Measures generation faithfulness independent of retrieval correctness.

These two metrics independently diagnose different failure modes:

- Low domain accuracy → retriever confused by cross-domain vocabulary overlap
- Low groundedness → generator hallucinating beyond retrieved context

---

## Discussion

### When to Use the Reranker

| Scenario | Recommendation |
| --- | --- |
| Single-domain homogeneous corpus | **Use reranker** — groundedness gains with no domain cost |
| Multi-domain heterogeneous corpus | **Ablate first** — domain accuracy may degrade |
| Faithfulness is the priority | **Use reranker** — groundedness improves |
| Retrieval domain precision is critical | **Bi-encoder only may dominate** |

### Limitations

- **Corpus size:** 20 papers; RL domain (6 papers) has high variance in domain accuracy estimates
- **Metric coverage:** Domain accuracy and groundedness only; full RAGAs (faithfulness, answer relevancy, context recall, context precision) not computed
- **Single reranker:** Only ms-marco-MiniLM evaluated; domain-adapted rerankers untested
- **Statistical power:** 25 questions; confidence intervals not reported; directional claims are preliminary

### Future Work

1. **Domain-aware reranking:** Fine-tune a cross-encoder on in-domain (query, passage) pairs to add domain coherence as a ranking signal
2. **Hybrid retrieval:** Reciprocal Rank Fusion over BM25 + dense retrieval — BM25 preserves domain-specific terminology more faithfully
3. **Query domain classification:** Classify query domain before retrieval, restrict search to corresponding document subset
4. **Full RAGAs evaluation:** Extend to four-metric benchmark (faithfulness, answer relevancy, context recall, context precision)
5. **Corpus scaling:** 50+ papers per domain to reduce minority-domain variance

---

## Reproducibility

### Install

```bash
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt
```

### Run

```bash
# Set GROQ_API_KEY in .env
python src/ingest.py       # Build vectorstore from PDFs
python src/evaluate.py     # Run evaluation
streamlit run app.py       # Launch UI
```

### Key Hyperparameters

| Parameter | Value |
| --- | --- |
| Embedding model | BAAI/bge-small-en-v1.5 (384-dim) |
| Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Stage-1 candidates | 20 |
| Final top-k passages | 4 |
| Chunk size | 512 characters |
| Chunk overlap | 50 characters |
| LLM | llama3-8b-8192 via Groq |
| Temperature | 0.1 |
| Eval set size | 25 questions |

---

## Project Structure

```
rag-project/
├── data/raw/              # 20 research PDFs (NLP, Transformers, RL)
├── chroma_db/             # Persisted vectorstore
├── src/
│   ├── ingest.py          # PDF ingestion pipeline
│   ├── retriever.py       # Two-stage retrieval
│   ├── generator.py       # Grounded LLM generation
│   ├── pipeline.py        # End-to-end orchestrator
│   └── evaluate.py        # Evaluation harness
├── notebooks/
│   └── ablation.py        # Reranker ablation
├── app.py                 # Streamlit UI
└── requirements.txt
```

---

## Live Demo

[**https://huggingface.co/spaces/pranav070/rag**](https://huggingface.co/spaces/pranav070/rag)

Ask any question about NLP, Transformer architectures, or Reinforcement Learning. The system retrieves relevant passages from 20 research papers and generates a grounded answer with full source citation.
