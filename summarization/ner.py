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

from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import pipeline
import numpy as np
import chromadb
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
from io import BytesIO
import spacy

# Load InLegalBERT model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBert")
model = AutoModel.from_pretrained("law-ai/InLegalBert").to(device)

# Load Indian Legal NER model
ner_model = AutoModelForTokenClassification.from_pretrained("MHGanainy/roberta-base-legal-multi-downstream-indian-ner").to(device)
ner_tokenizer = AutoTokenizer.from_pretrained("MHGanainy/roberta-base-legal-multi-downstream-indian-ner")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

# Load spaCy transformer-based model
spacy_nlp = spacy.load("en_core_web_trf")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to perform Named Entity Recognition (NER)
def extract_named_entities(text):
    ner_results = ner_pipeline(text)
    named_entities = {}
    for ent in ner_results:
        label = ent["entity_group"]
        if label not in named_entities:
            named_entities[label] = []
        named_entities[label].append(ent["word"])
    for key in named_entities:
        named_entities[key] = list(set(named_entities[key]))
    return named_entities

# Function to perform spaCy NER
def extract_named_entities_spacy(text):
    doc = spacy_nlp(text)
    named_entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in named_entities:
            named_entities[label] = []
        named_entities[label].append(ent.text)
    for key in named_entities:
        named_entities[key] = list(set(named_entities[key]))
    return named_entities


def compare_ner_models(text):
    legal_ner_entities = extract_named_entities(text)
    spacy_entities = extract_named_entities_spacy(text)

    legal_df = pd.DataFrame(
        [(label, ent) for label, ents in legal_ner_entities.items() for ent in ents],
        columns=["Entity Label", "Entity"]
    )

    spacy_df = pd.DataFrame(
        [(label, ent) for label, ents in spacy_entities.items() for ent in ents],
        columns=["Entity Label", "Entity"]
    )

    return legal_df, spacy_df



# Example usage (without Streamlit)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Legal Case Analyzer - NER Comparison Only")
    parser.add_argument("--file", type=str, required=True, help="Path to the legal case PDF")
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        case_text = extract_text_from_pdf(f)
    compare_ner_models(case_text)
