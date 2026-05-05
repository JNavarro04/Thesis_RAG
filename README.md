# Thesis_RAG — BSc AI Thesis (in progress)

**Comparing Fixed-size vs. Semantic RAG Chunking Strategies in a Virtual Human Supermarket Guide**

Radboud University · BSc Artificial Intelligence · Expected completion: July 2026

---

## Overview

This project implements and evaluates two Retrieval-Augmented Generation (RAG) chunking strategies for a Unity 3D virtual human supermarket guide. The system answers product and navigation questions by retrieving relevant chunks from a knowledge base and generating responses via an LLM.

The Unity environment was provided by the Theis Supervisor. Expect files to be irrelevant until final completion + upload.

## Research Question

Does semantic chunking (Kiss et al., 2025) outperform fixed-size chunking in terms of accuracy, faithfulness, and latency in a RAG-based virtual human system?

## Tech Stack

- **Language:** Python
- **LLM:** GPT (OpenAI API)
- **Vector store:** ChromaDB
- **Evaluation:** RAGAS + custom metrics
- **Virtual Human:** Unity 3D
- **Speech:** Google Cloud Speech-to-Text

## Evaluation Metrics

- Recall@k, MRR@k
- Hallucination rate
- Citation correctness
- Indexing / retrieval / generation latency
- Statistical tests: McNemar, Wilcoxon signed-rank, bootstrap CI

## Project Structure

```
pipeline/        # Chunking and indexing logic
evaluation/      # Retrieval and generation evaluation scripts
data/            # Knowledge base and QA dataset
usecase/         # Virtual human integration
```

---

_This repository is part of an ongoing BSc thesis. Results and full documentation will be added upon completion._
