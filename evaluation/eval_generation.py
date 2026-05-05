# eval_generation.py
#
# Generation evaluation for both chunking strategies.
#
# For each of the 150 QA questions this script:
#   1. Retrieves the top-5 most relevant chunks from ChromaDB
#   2. Sends those chunks + the question to GPT-4o-mini to generate an answer
#   3. Generates a second answer WITHOUT context (parametric knowledge only)
#   4. Asks GPT-4o-mini to score the answer 0-3 and flag hallucinations
#   5. Scores with four RAGAS metrics
#   6. Measures citation faithfulness: how much the context changed the answer
#      (low similarity = model genuinely used context; high = post-rationalisation)
#      Based on Wallat et al. (2025)
#
# Runs twice — once for fixed-size chunks, once for semantic chunks.
# Everything else (model, prompt, k) is identical between the two runs.
#
# Output: evaluation/results/results_generation.csv

import json
import time
import csv
import sys
import asyncio
import numpy as np
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
)

# ── Paths and settings ────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config_claude import get_openai_key

OPENAI_API_KEY  = get_openai_key()
QA_PATH         = ROOT / "data" / "qa_dataset.json"
CHROMA_DB_PATH  = ROOT / "chroma_db"
OUTPUT_PATH     = ROOT / "evaluation" / "results" / "results_generation.csv"
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_MODEL       = "gpt-4o-mini"
K_RETRIEVE      = 5
COLLECTIONS     = ["supermarket_fixed", "supermarket_semantic"]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a virtual human guide for a supermarket. "
    "Answer the customer's question using only the information in the context. "
    "Be concise and accurate. If the answer is not in the context, say so explicitly."
)

RUBRIC_PROMPT = """\
You are evaluating a RAG system answer against a ground truth answer.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}
Context used: {context}

Score 0-3:
3 = Fully correct and complete
2 = Mostly correct, minor omission
1 = Partially correct, significant gap
0 = Wrong or hallucinated

Also check if the answer contains claims NOT supported by the context.

Respond ONLY in this JSON format:
{{"score": <0-3>, "hallucination": <true/false>, "reasoning": "<one sentence>"}}"""


# ── Helper functions ──────────────────────────────────────────────────────────

def retrieve_chunks(collection, embedding_model, question):
    """Return the top-K chunk texts most similar to the question."""
    question_embedding = embedding_model.encode(question, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=K_RETRIEVE,
        include=["documents"],
    )
    return results["documents"][0]


def generate_answer(openai_client, question, chunk_texts):
    """Send the question and retrieved chunks to the LLM and return the answer."""
    context = "\n\n---\n\n".join(chunk_texts)
    t0 = time.perf_counter()
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    latency = time.perf_counter() - t0
    answer  = response.choices[0].message.content.strip()
    return answer, latency


def judge_answer(openai_client, question, answer, ground_truth, context):
    """Ask the LLM to score the answer 0-3 and flag any hallucinations."""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            context=context,
        )}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        result        = json.loads(response.choices[0].message.content)
        rubric_score  = int(result.get("score", 0))
        hallucination = bool(result.get("hallucination", False))
        reasoning     = result.get("reasoning", "")
        return rubric_score, hallucination, reasoning
    except Exception:
        return 0, False, "parse error"


def generate_without_context(openai_client, question):
    """
    Generate an answer using only the LLM's parametric knowledge — no retrieved context.
    Used to measure citation faithfulness: if this answer is very similar to the
    answer generated WITH context, the model was not genuinely using the context
    (post-rationalisation). Based on Wallat et al. (2025).
    """
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a supermarket guide. Answer the question as best you can."},
            {"role": "user",   "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def citation_faithfulness(embedding_model, answer_with_context, answer_without_context):
    """
    Measure how much the retrieved context changed the model's answer.
    Computes cosine similarity between the two answers (with vs without context).

    Interpretation (Wallat et al., 2025):
      Low similarity  → model genuinely used the context (faithful)
      High similarity → model gave the same answer regardless of context
                        (post-rationalisation, not faithful)

    Returns a score between 0 and 1.
    """
    emb_with    = embedding_model.encode(answer_with_context,    convert_to_numpy=True)
    emb_without = embedding_model.encode(answer_without_context, convert_to_numpy=True)
    similarity  = float(np.dot(emb_with, emb_without) /
                        (np.linalg.norm(emb_with) * np.linalg.norm(emb_without)))
    return round(similarity, 4)


def score_with_ragas(ragas_metrics, question, answer, chunk_texts, ground_truth):
    """
    Score one question-answer pair with four RAGAS metrics.
    RAGAS metrics are async so we use asyncio.run() to call them from normal code.
    Returns a dict with one score per metric.
    """
    async def run_all():
        results = {}

        try:
            r = await ragas_metrics["faithfulness"].ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=chunk_texts,
            )
            results["faithfulness"] = round(float(r.value), 4)
        except Exception:
            results["faithfulness"] = 0.0

        try:
            r = await ragas_metrics["answer_relevancy"].ascore(
                user_input=question,
                response=answer,
            )
            results["answer_relevancy"] = round(float(r.value), 4)
        except Exception:
            results["answer_relevancy"] = 0.0

        try:
            r = await ragas_metrics["context_precision"].ascore(
                user_input=question,
                reference=ground_truth,
                retrieved_contexts=chunk_texts,
            )
            results["context_precision"] = round(float(r.value), 4)
        except Exception:
            results["context_precision"] = 0.0

        try:
            r = await ragas_metrics["context_recall"].ascore(
                user_input=question,
                retrieved_contexts=chunk_texts,
                reference=ground_truth,
            )
            results["context_recall"] = round(float(r.value), 4)
        except Exception:
            results["context_recall"] = 0.0

        return results

    return asyncio.run(run_all())


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Load questions
    questions = json.loads(QA_PATH.read_text(encoding="utf-8"))["questions"]
    print(f"Loaded {len(questions)} questions.")

    # Initialise components shared across both collections
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    openai_client   = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client   = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # RAGAS metrics need the async OpenAI client
    async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
    ragas_llm    = llm_factory(model=LLM_MODEL, client=async_openai)
    ragas_emb    = OpenAIEmbeddings(client=async_openai)
    ragas_metrics = {
        "faithfulness":      Faithfulness(llm=ragas_llm),
        "answer_relevancy":  AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall":    ContextRecall(llm=ragas_llm),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for collection_name in COLLECTIONS:
        method     = collection_name.split("_")[1]  # "fixed" or "semantic"
        collection = chroma_client.get_collection(collection_name)
        print(f"\n{'='*50}")
        print(f"Evaluating: {collection_name} ({collection.count()} chunks)")
        print(f"{'='*50}")

        for i, q in enumerate(questions, start=1):
            question     = q["question"]
            ground_truth = q["ground_truth_answer"]

            # Step 1: retrieve
            chunk_texts = retrieve_chunks(collection, embedding_model, question)
            context     = "\n\n---\n\n".join(chunk_texts)

            # Step 2: generate answer WITH context
            answer, latency = generate_answer(openai_client, question, chunk_texts)

            # Step 3: generate answer WITHOUT context and measure citation faithfulness
            # (only needs to be done once per question, not per collection — but we
            # run it per collection so results are fully independent)
            answer_no_context  = generate_without_context(openai_client, question)
            cite_faithfulness  = citation_faithfulness(embedding_model, answer, answer_no_context)

            # Step 4: judge
            score, hallucination, reasoning = judge_answer(
                openai_client, question, answer, ground_truth, context
            )

            # Step 5: RAGAS
            ragas_scores = score_with_ragas(
                ragas_metrics, question, answer, chunk_texts, ground_truth
            )

            print(f"  [{i:3d}/150] {q['id']} | "
                  f"score={score} | hallucination={hallucination} | {latency:.2f}s")

            all_rows.append({
                "question_id":           q["id"],
                "category":              q["category"],
                "method":                method,
                "answer":                answer,
                "rubric_score":          score,
                "hallucination":         int(hallucination),
                "gen_latency_s":         round(latency, 4),
                "judge_reasoning":       reasoning,
                "citation_faithfulness": cite_faithfulness,
                "faithfulness":          ragas_scores["faithfulness"],
                "answer_relevancy":      ragas_scores["answer_relevancy"],
                "context_precision":     ragas_scores["context_precision"],
                "context_recall":        ragas_scores["context_recall"],
            })

    # Save results
    fieldnames = [
        "question_id", "category", "method",
        "rubric_score", "hallucination", "gen_latency_s",
        "citation_faithfulness",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "judge_reasoning", "answer",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_PATH}")
    print_summary(all_rows)


def print_summary(rows):
    print("\n" + "=" * 60)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 60)
    metrics = [
        "rubric_score", "hallucination", "gen_latency_s",
        "citation_faithfulness",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ]
    for method in ["fixed", "semantic"]:
        method_rows = [r for r in rows if r["method"] == method]
        n = len(method_rows)
        print(f"\n{method.upper()} — {n} questions")
        for m in metrics:
            mean = sum(r[m] for r in method_rows) / n
            print(f"  {m:<25} {mean:.4f}")
        print("  Per-category rubric score:")
        for cat in sorted(set(r["category"] for r in method_rows)):
            cat_rows = [r for r in method_rows if r["category"] == cat]
            mean     = sum(r["rubric_score"] for r in cat_rows) / len(cat_rows)
            print(f"    {cat:<30} {mean:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()