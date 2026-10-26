import os
import time
from pathlib import Path
import torch
import chromadb
import streamlit as st
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "law-ai/InLegalBERT"
GEMINI_MODEL = "gemini-3.8-flash"

# This file is assumed to be inside QnA_model/
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"


# ==========================================
# LOAD INLEGALBERT ONCE PER STREAMLIT SESSION
# ==========================================
@st.cache_resource
def load_embedding_model():
    # Use the cached model files; avoid repeated Hugging Face retries.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    embedding_model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    embedding_model.eval()
    return tokenizer, embedding_model


tokenizer, embedding_model = load_embedding_model()


# ==========================================
# CHROMADB
# ==========================================
@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name="legal_qna")


collection = load_collection()


# ==========================================
# GEMINI CONFIGURATION
# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


# ==========================================
# EMBEDDINGS
# ==========================================
def get_embedding(text):
    """Generate an attention-mask-aware InLegalBERT embedding."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        outputs = embedding_model(**inputs)
        token_embeddings = outputs.last_hidden_state

        # Exclude padding tokens from mean pooling.
        mask = inputs["attention_mask"].unsqueeze(-1).to(
            token_embeddings.dtype
        )

        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embedding = (summed / counts).squeeze(0)

    return embedding.cpu().numpy().tolist()


# ==========================================
# RETRIEVE LEGAL CONTEXT
# ==========================================
def retrieve_context(query, top_k=3):
    """Retrieve stored answers and only fall back when none are found."""
    query_embedding = get_embedding(query)

    answer_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where={"type": "answer"},
        include=["metadatas"],
    ) if collection.count() > 0 else {"metadatas": [[]]}

    contexts = []

    for metadata in (answer_results.get("metadatas") or [[]])[0]:
        if metadata and metadata.get("answer"):
            contexts.append(str(metadata["answer"]))

    # Search question records only if no usable answer was found.
    if not contexts and collection.count() > 0:
        question_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where={"type": "question"},
            include=["metadatas"],
        )

        for metadata in (question_results.get("metadatas") or [[]])[0]:
            if metadata and metadata.get("answer"):
                answer = str(metadata["answer"])
                if answer not in contexts:
                    contexts.append(answer)

    return "\n\n".join(contexts)


# ==========================================
# ANSWER LEGAL QUESTIONS
# ==========================================
def answer_question(query):
    """Generate a concise answer and report timing in the terminal."""
    query = query.strip()

    if not query:
        return "Please enter a legal question."

    if not API_KEY:
        return (
            "Gemini API key is missing. Set the GEMINI_API_KEY "
            "environment variable and restart Streamlit."
        )

    total_start = time.perf_counter()

    try:
        context = retrieve_context(query)
        retrieval_end = time.perf_counter()

        if context:
            legal_text = context[:6000]
        else:
            legal_text = (
                "No relevant text was retrieved from the legal database."
            )

        prompt = f"""
You provide general legal information about Indian law.

Answer the question clearly and concisely.
- Use retrieved legal text when relevant.
- Do not invent legal sections, case citations, or punishments.
- If the available information is insufficient, say so.
- Distinguish general legal knowledge from information in the
  retrieved text.
- Mention uncertainty where appropriate.
- This is general information, not a substitute for legal advice.

Retrieved legal text:
{legal_text}

Question:
{query}

Answer:
"""

        gemini_model = genai.GenerativeModel(GEMINI_MODEL)

        generation_start = time.perf_counter()
        response = gemini_model.generate_content(prompt)
        finished = time.perf_counter()

        answer = getattr(response, "text", None)

        print("\n========== LEGAL Q&A TIMING ==========")
        print(f"Retrieval: {retrieval_end - total_start:.2f} seconds")
        print(f"Gemini:    {finished - generation_start:.2f} seconds")
        print(f"Total:     {finished - total_start:.2f} seconds")
        print("======================================\n")

        if not answer or not answer.strip():
            return "The model returned an empty answer. Please try again."

        return answer.strip()

    except Exception as exc:
        print(f"Legal Q&A error: {type(exc).__name__}: {exc}")
        return (
            "Sorry, I couldn't generate an answer. "
            "Check the terminal for the error details."
        )


# ==========================================
# OPTIONAL COMMAND-LINE TEST
# ==========================================
if __name__ == "__main__":
    question = input("Enter your legal question: ")
    print("Answer:", answer_question(question))
