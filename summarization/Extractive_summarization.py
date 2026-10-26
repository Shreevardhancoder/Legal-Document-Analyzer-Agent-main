import sys
import types
import os
import json
import asyncio
import pandas as pd

# Fix for torch.classes error in some environments
import torch
if not hasattr(torch, "_classes"):
    torch._classes = types.SimpleNamespace()
if not hasattr(torch._classes, "__path"):
    torch._classes.__path__ = []

from transformers import AutoTokenizer, AutoModel
import numpy as np
import chromadb
from nltk.tokenize import sent_tokenize
import PyPDF2

# Load InLegalBERT model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBert")
model = AutoModel.from_pretrained("law-ai/InLegalBert").to(device)

# Initialize ChromaDB client
client_chroma = chromadb.PersistentClient(path="../chroma_db")
legal_docs_collection = client_chroma.get_or_create_collection("legal_docs")

# Function to get embeddings
def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding / np.linalg.norm(cls_embedding)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to extract batch metadata
def extract_case_metadata(case_id):
    try:
        parts = case_id.split("_")
        batch_number = int(parts[1])
        index_in_batch = int(parts[2])
        parquet_path = f"../processed_batch_{batch_number}.parquet"
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if index_in_batch < len(df):
                row = df.iloc[index_in_batch]
                metadata_str = (
                    f"Court: {row.get('Court_Name_Normalized', 'Unknown')}\n"
                    f"Case Type: {row.get('Case_Type', 'Unknown')}\n"
                    f"Doc URL: {row.get('Doc_url', 'Unknown')}"
                )
                return metadata_str
        return "Metadata not available"
    except Exception as e:
        return f"Metadata error: {str(e)}"

# Function to summarize a single uploaded case (extractive)
def summarize_uploaded_case(uploaded_text, top_n=8):
    sentences = sent_tokenize(uploaded_text)
    embeddings = [get_embedding(sent) for sent in sentences]
    query_vec = get_embedding(uploaded_text)
    scores = [np.dot(query_vec, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [sentences[i].strip() for i in top_indices]

# Function for extractive summarization using top relevant cases
def summarize_case_using_top_cases(uploaded_case_text, uploaded_file_name, top_k_cases=3, top_n_sentences=5):
    # Get embedding for uploaded case
    query_embedding = get_embedding(uploaded_case_text)
    
    # Query similar cases
    results = legal_docs_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k_cases + 3
    )
    retrieved_ids = results.get("ids", [[]])[0] or []
    
    # Filter out uploaded file's ID
    retrieved_ids = [case_id for case_id in retrieved_ids if not uploaded_file_name.lower() in case_id.lower()]
    related_case_ids = retrieved_ids[:top_k_cases]
    
    # Process uploaded case
    uploaded_summary = summarize_uploaded_case(uploaded_case_text)
    
    # Process related cases
    related_cases_data = []
    related_sentences = []
    
    for case_id in related_case_ids:
        # Get case data
        case_data = legal_docs_collection.get(ids=[case_id])
        case_texts = case_data.get("documents", []) if case_data else []
        
        # Get metadata
        metadata = extract_case_metadata(case_id)
        
        # Store case data
        related_cases_data.append({
            "case_id": case_id,
            "metadata": metadata
        })
        
        # Process sentences
        for text in case_texts:
            sentences = sent_tokenize(text)
            for sent in sentences:
                emb = get_embedding(sent)
                score = np.dot(query_embedding, emb)
                related_sentences.append({
                    "sentence": sent.strip(),
                    "case_id": case_id,
                    "score": score
                })
    
    # Sort and select top related sentences
    related_sentences.sort(key=lambda x: x["score"], reverse=True)
    top_related_sentences = related_sentences[:top_n_sentences]
    
    # Prepare final summary
    summary = {
        "uploaded_case": {
            "key_points": [
                {"point_number": i + 1, "text": sent}
                for i, sent in enumerate(uploaded_summary)
            ]
        },
        "related_cases": {
            "metadata": [
                {
                    "case_id": data["case_id"],
                    "details": data["metadata"]
                }
                for data in related_cases_data
            ],
            "relevant_points": [
                {
                    "point_number": i + 1,
                    "text": sent["sentence"],
                    "source_case": sent["case_id"]
                }
                for i, sent in enumerate(top_related_sentences)
            ]
        }
    }
    
    return summary


# Example usage (without Streamlit)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Legal Case Summarization")
    parser.add_argument("--file", type=str, required=True, help="Path to the legal case PDF")
    args = parser.parse_args()

    uploaded_file_name = os.path.basename(args.file).strip()

    with open(args.file, "rb") as f:
        case_text = extract_text_from_pdf(f)

    summary = summarize_case_using_top_cases(case_text, uploaded_file_name)

    print("\n📄 Final Extractive Summary:")
    print(summary)
