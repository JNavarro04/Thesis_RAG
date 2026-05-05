# use case for supermarket vh environment
# OpenAI as LLM

import os

import experiment_settings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import csv
import smtplib
from datetime import datetime
from email.message import EmailMessage
import sys

import openai
import speechToUnity
import utils
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import config_claude  # keep using existing config, or rename to config_openai

log_data = []
os.environ["HF_TOKEN"] = config_claude.get_huggingface_token()


# Log speaker and spoken text
def log_string(speaker: str, spoken_text: str):
    """Sla string + tijdstip op in dictionary."""
    current_time_log = datetime.now().isoformat(timespec="milliseconds")
    log_data.append((current_time_log, speaker, spoken_text))


# Save log to csv file for sending by email
def save_log_to_csv(filename="log.csv"):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "speaker", "spoken text"])
        for log_time, speaker, spoken_text in log_data:
            writer.writerow([log_time, speaker, spoken_text])


# Send email with attachment
def send_email_with_attachment(
    sender_email, app_password, receiver_email, subject, body, attachment_path
):
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = attachment_path

    msg.add_attachment(file_data, maintype="text", subtype="csv", filename=file_name)

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)


# ---------------------------------------------------
# Step 1 - Load chunks and embeddings from JSON
# ---------------------------------------------------
def load_enriched_json(filepath: str) -> tuple[list[dict], list[np.ndarray]]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]
    qa_pairs = [
        {
            "question":              item["question"],
            "answer":                item["answer"],
            "alternative_questions": item.get("alternative_questions", []),
        }
        for item in items
    ]
    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in items]
    return qa_pairs, embeddings


# ---------------------------------------------------
# Step 4 - Retrieval
# ---------------------------------------------------
def retrieve(query_retrieve, model_retrieve, qa_pairs, embeddings, top_k_retrieve=3):
    query_emb = model_retrieve.encode([query_retrieve])[0]

    scores = []
    for i, emb in enumerate(embeddings):
        score = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
        scores.append((i, qa_pairs[i], score))

    ranked = sorted(scores, key=lambda x: x[2], reverse=True)
    return ranked[:top_k_retrieve]


# ---------------------------------------------------
# Step 5 - Evaluate retrieval
# ---------------------------------------------------
def evaluate_chunk(query_evalchunk, query_emb, chunk_dict, chunk_emb):
    cosine = float(
        np.dot(query_emb, chunk_emb)
        / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
    )

    query_tokens = set(query_evalchunk.lower().split())
    chunk_text = f"{chunk_dict['question']} {chunk_dict['answer']}"
    chunk_tokens = set(chunk_text.lower().split())

    overlap = query_tokens.intersection(chunk_tokens)
    overlap_count = len(overlap)
    overlap_pct = overlap_count / len(query_tokens) if query_tokens else 0

    return {
        "cosine_similarity": cosine,
        "overlap_tokens": sorted(overlap),
        "overlap_count": overlap_count,
        "overlap_percentage": overlap_pct,
    }


def evaluate_retrieval(query_evalretrieval, embedding_model, qa_pairs, embeddings, top_k_evalretrieval=3):
    query_emb = embedding_model.encode([query_evalretrieval], convert_to_numpy=True)[0]

    results = []
    for i, (pair, chunk_emb) in enumerate(zip(qa_pairs, embeddings)):
        evaluation_metrics = evaluate_chunk(query_evalretrieval, query_emb, pair, chunk_emb)
        results.append({"index": i, "pair": pair, **evaluation_metrics})

    ranked = sorted(results, key=lambda x: x["cosine_similarity"], reverse=True)

    for rank, r in enumerate(ranked[:top_k_evalretrieval], start=1):
        log_string("top_k_retrieval_ranking", f"\nRank {rank} | Chunk #{r['index']}")
        log_string("top_k_retrieval_ranking", f"Cosine similarity : {r['cosine_similarity']:.4f}")
        log_string("top_k_retrieval_ranking", f"Overlap count     : {r['overlap_count']}")
        log_string("top_k_retrieval_ranking", f"Overlap %         : {r['overlap_percentage']:.2%}")
        log_string("top_k_retrieval_ranking", f"Overlap tokens    : {r['overlap_tokens']}")
        log_string("top_k_retrieval_ranking", format_chunk_preview(r["pair"], max_chars=200))
    return ranked


def format_chunk_preview(chunk: dict, max_chars: int = 200) -> str:
    q = chunk.get("question", "")
    a = chunk.get("answer", "")
    f_text = f"Q: {q}\nA: {a}"
    if len(f_text) > max_chars:
        return f_text[:max_chars] + "..."
    return f_text


# ---------------------------------------------------
# Step 7 - Evaluate LLM response
# ---------------------------------------------------
def answer_context_similarity(llm_answer, llm_context, embedding_model):
    emb = embedding_model.encode([llm_answer, llm_context], convert_to_numpy=True)
    log_string("answer_context_similarity", str(cosine_similarity([emb[0]], [emb[1]])[0][0]))
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def answer_query_similarity(answer_similarity, query_similarity, embedding_model):
    emb = embedding_model.encode([answer_similarity, query_similarity], convert_to_numpy=True)
    log_string("answer_query_similarity", str(cosine_similarity([emb[0]], [emb[1]])[0][0]))
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def answer_chunk_overlap(answer_overlap, chunk_dicts):
    answer_tokens = set(answer_overlap.lower().split())
    chunk_tokens = set(
        " ".join(
            f"{c['question']} {c['answer']}"
            for c in chunk_dicts
        ).lower().split()
    )
    overlap = answer_tokens.intersection(chunk_tokens)
    overlap_data = {
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / len(answer_tokens) if answer_tokens else 0,
        "overlap_tokens": sorted(overlap)
    }
    overlap_count = str(len(overlap))
    overlap_pct = str(len(overlap)/len(answer_tokens))
    overlap_tokens = str(sorted(overlap))
    log_string("overlap_count", overlap_count)
    log_string("overlap_pct", overlap_pct)
    log_string("overlap_tokens", overlap_tokens)
    return overlap_data


def evaluate_llm_answer(query_evaluation, answer_evaluation, top_chunks_evaluation, embedding_model):
    context_evaluation = "\n".join(
        f"{c['question']} {c['answer']}"
        for c in top_chunks_evaluation
    )
    llm_answer_evaluation = {
        "answer_query_similarity": answer_query_similarity(
            answer_evaluation, query_evaluation, embedding_model
        ),
        "answer_context_similarity": answer_context_similarity(
            answer_evaluation, context_evaluation, embedding_model
        ),
        **answer_chunk_overlap(answer_evaluation, top_chunks_evaluation)
    }
    return llm_answer_evaluation


# Strip markdown from LLM responses
def strip_markdown(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s,.!?;:()\'\"\-]', '', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# Build system prompt
def llm_role_function(query, context_for_llm, conversation_history):
    history_text = "\n".join(
        f"{'Klant' if m['role'] == 'user' else 'Anna'}: {m['content']}"
        for m in conversation_history
    ) if conversation_history else "Geen eerdere berichten."

    llm_role = (
        f"You are a helpful digital legal professional. "
        f"A person has received a summons from the subdistrict court judge. "
        f"This person asks you a question {query} "
        f"related to how to prepare for the subdistrict court hearing. "
        f"Only provide general, procedural information about preparing for this hearing "
        f"before the subdistrict court. In the first instance, base yourself on the "
        f"context {context_for_llm} provided. When questions arise about information that is not in "
        f"this knowledge source, use the Large Language Model. Don't give predictions about guilt, "
        f"punishment or outcome. And don't make any definitive statements. Give pure advice "
        f"and information. Do not give strategic defense advice. "
        f"Previous conversation:\n{history_text}\n"
        f"Stay neutral, friendly, and clear. Talk in Dutch and use simple language and short sentences. "
        f"Briefly explain what the answer means in practical terms for preparing for the hearing."
        f"Use a maximum of five sentences."
        f"IMPORTANT FORMATTING RULES - you must follow these strictly: "
        f"Write in plain flowing prose only. Do not use bullet points, numbered lists, or dashes. "
        f"Do not use headers or section titles. Do not use horizontal dividers like --- or ***. "
        f"Do not use markdown formatting of any kind. "
        f"Do not use emoticons or emoji. "
        f"Write everything as natural, connected sentences that can be read aloud fluently. "
        f"The response will be spoken out loud by a text-to-speech system, so it must sound "
        f"like natural spoken Dutch with no symbols, signs, or formatting."
    )
    return llm_role


# -------------------------------------------------------
# Call OpenAI API  (replaces answer_question for Claude)
# -------------------------------------------------------
def answer_question(client, transcript, llm_role, conversation_history, messages):
    try:
        messages.append({"role": "user", "content": transcript})

        # OpenAI expects the system prompt as the first message in the list
        full_messages = [{"role": "system", "content": llm_role}] + messages

        response = client.chat.completions.create(
            model="gpt-4o",          # change to "gpt-4o-mini" for cheaper / faster
            max_tokens=1024,
            messages=full_messages
        )
        return response

    except openai.BadRequestError:
        language = experiment_settings.experiment_language()
        gender = experiment_settings.experiment_gender()
        speechToUnity.say_something("De service is overladen of nog niet klaar.", language, gender)
        return None


# ---------------------------------------------------
# Main dialogue function
# ---------------------------------------------------
def dialogue_legal_advise():
    enriched_json = "jurloket_enriched.json"
    embed_model   = "sentence-transformers/all-mpnet-base-v2"
    top_k         = 3
    log_string("Large Language Model: ", "OpenAI")

    print(f"Loading chunks and embeddings from {enriched_json}...")
    chunks_in_json_format, chunk_embeddings = load_enriched_json(enriched_json)
    print(f"Loaded {len(chunks_in_json_format)} Q&A pairs.")

    print(f"Loading embedding model: {embed_model}...")
    model = SentenceTransformer(embed_model)

    language = experiment_settings.experiment_language()
    gender   = experiment_settings.experiment_gender()

    # ---- OpenAI client (replaces anthropic.Anthropic) ----
    api_key = config_claude.get_openai_key()   # add this method to config_claude,
                                                # or replace with: os.environ["OPENAI_API_KEY"]
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    # ------------------------------------------------------

    avatar_name = "Anna"
    user_name   = "Klant"


    speech_recognizer = utils.init_ms_azure_stt("en-GB")

    welcome_statement = "Hello. how can I help"
    print("Anna:> " + welcome_statement)
    log_string(avatar_name, welcome_statement)
    speechToUnity.say_something(welcome_statement, language, gender)
    while speechToUnity.is_speaking:
        time.sleep(0.1)

    end = False
    time_out: int = 0
    conversation_history = []
    messages = []

    while not end:
        result = speech_recognizer.recognize_once()
        query  = result.text

        if not query:
            time_out += 1
            if time_out >= 4:
                end = True
            continue

        print("Query: " + query)
        speechToUnity.send_chat("Klant", query)
        log_string(user_name, query)

        if re.search(re.escape("bye"), query, re.IGNORECASE):
            speechToUnity.send_chat("Klant", query)
            break

        ranked_results = retrieve(query, model, chunks_in_json_format, chunk_embeddings, top_k)
        evaluate_retrieval(query, model, chunks_in_json_format, chunk_embeddings)
        top_chunks = [pair for _, pair, _ in ranked_results]

        context = "\n\n---\n\n".join(
            f"Q: {c['question']}\nA: {c['answer']}"
            for c in top_chunks
        )

        llm_role = llm_role_function(query, context, conversation_history)
        messages = []

        try:
            response = answer_question(client, query, llm_role, conversation_history, messages)
            if response is not None:
                # ---- OpenAI response extraction (replaces response.content[0].text) ----
                response_text = response.choices[0].message.content
                # -------------------------------------------------------------------------
                response_text = strip_markdown(response_text)
                print("Anna:> " + response_text)
                log_string(avatar_name, response_text)
                speechToUnity.say_something(response_text, language, gender)
                while speechToUnity.is_speaking:
                    time.sleep(0.1)

                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": response_text})

        except openai.BadRequestError:
            speechToUnity.say_something("De service is overladen of nog niet klaar.", language, gender)


    if time_out < 4:
        text = "Thanks see you next time!"
    else:
        text = "I did not hear you. I guess you left goodbye."

    log_string(avatar_name, text)
    speechToUnity.say_something(text,language,gender)
    while speechToUnity.is_speaking:
        time.sleep(0.1)

    csv_file = "log.csv"
    save_log_to_csv(csv_file)

    send_email_with_attachment(
        sender_email="jain.understanding@gmail.com",
        app_password="jcgh ngpq vtge ycuh",
        receiver_email="juan.navarroarias-salgado@ru.nl",
        subject="Programma log",
        body="Dear Juan, \nSee attachment for logs.\n",
        attachment_path=csv_file
    )