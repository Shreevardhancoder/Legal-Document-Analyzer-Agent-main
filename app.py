import sys
import types
import os
import json
import asyncio
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import chromadb
from transformers import AutoTokenizer, AutoModel
import PyPDF2
import streamlit as st
import torch

# Fix for torch.classes error
if not hasattr(torch, "_classes"):
    torch._classes = types.SimpleNamespace()
if not hasattr(torch._classes, "__path__"):
    torch._classes.__path__ = []

# Fix for asyncio event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Load InLegalBERT model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBert")
model = AutoModel.from_pretrained("law-ai/InLegalBert").to(device)

# Initialize ChromaDB client
chroma_db_path = os.path.join(current_dir, "chroma_db")
client_chroma = chromadb.PersistentClient(path=chroma_db_path)
legal_docs_collection = client_chroma.get_or_create_collection("legal_docs")

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding / np.linalg.norm(cls_embedding)

try:
    from QnA_model.qna import answer_question
except ImportError:
    from QnA_model.qna import answer_question  # Try alternative import

from summarization.abstractive_summarization import extract_text_from_pdf, chunk_text_with_overlap, summarize_chunks
from summarization.ner import compare_ner_models


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
    """Extract metadata from case ID using parquet files"""
    try:
        parts = case_id.split("_")
        batch_number = int(parts[1])
        index_in_batch = int(parts[2])
        parquet_path = f"./processed_batch_{batch_number}.parquet"
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if index_in_batch < len(df):
                row = df.iloc[index_in_batch]
                # Return metadata as a dictionary for flexible formatting
                return {
                    "court": row.get('Court_Name_Normalized', 'Unknown'),
                    "case_type": row.get('Case_Type', 'Unknown'),
                    "doc_url": row.get('Doc_url', '#')
                }
        return None
    except Exception as e:
        return None

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

def main():
    st.title("📚 Legal Document Analysis System")
    
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for functionality selection
    st.sidebar.title("Choose Functionality")
    functionality = st.sidebar.radio(
        "Select what you want to do:",
        ["Extractive Summarization", 
         "Abstractive Summarization",
         "Named Entity Recognition",
         "Legal Q&A"]
    )
    
    if functionality == "Legal Q&A":
        st.header("❓ Legal Questions & Answers")
        
        # Display chat history with unique keys
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.container():
                st.text_area(
                    "Question:", 
                    value=q, 
                    height=70, 
                    disabled=True,
                    key=f"question_{i}"
                )
                st.text_area(
                    "Answer:", 
                    value=a, 
                    height=150, 
                    disabled=True,
                    key=f"answer_{i}"
                )
                st.markdown("---")
        
        # Input for new question
        question = st.text_input("Enter your legal question:", key="new_question")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Get Answer", key="get_answer"):
                if question:
                    with st.spinner("Finding answer..."):
                        answer = answer_question(question)
                        st.session_state.chat_history.append((question, answer))
                        st.success("Answer found!")
                        st.rerun()
        
        with col2:
            if st.button("Clear History", key="clear_history"):
                st.session_state.chat_history = []
                st.rerun()

    elif functionality == "Extractive Summarization":
        st.header("📝 Extractive Summarization")
        uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf", key="extractive_uploader")
        
        if uploaded_file:
            try:
                with st.spinner("Processing document..."):
                    case_text = extract_text_from_pdf(uploaded_file)
                    st.info("Document uploaded successfully!")
                    
                    if st.button("Generate Summary", key="extractive_summary_btn"):
                        with st.spinner("Generating extractive summary..."):
                            try:
                                summary = summarize_case_using_top_cases(case_text, uploaded_file.name)
                                print(summary)
                                # Display uploaded case summary
                                st.subheader("📄 Key Points from Uploaded Document")
                                for point in summary["uploaded_case"]["key_points"]:
                                    st.write(f"{point['point_number']}. {point['text']}")
                                
                                # Display related cases
                                st.subheader("📚 Related Cases")
                                for case in summary["related_cases"]["metadata"]:
                                    with st.expander(f"Case ID: {case['case_id']}", expanded=True):
                                        metadata = case["details"]
                                        if metadata:
                                            st.write(f"**Court:** {metadata['court']}")
                                            st.write(f"**Case Type:** {metadata['case_type']}")
                                            st.markdown(f"**Doc URL:** [{metadata['doc_url']}]({metadata['doc_url']})")
                                        else:
                                            st.write("Metadata not available")
                                
                                # Display relevant points from related cases
                                st.subheader("🔍 Relevant Points from Related Cases")
                                for point in summary["related_cases"]["relevant_points"]:
                                    with st.expander(f"Point {point['point_number']}", expanded=True):
                                        st.write(f"**Source:** Case {point['source_case']}")
                                        st.write(point["text"])
                                
                                st.success("Summary generated successfully!")
                                
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                                st.info("Please try again or contact support if the issue persists.")
                                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.info("Please ensure the uploaded file is a valid PDF document.")




    
    elif functionality == "Abstractive Summarization":
        st.header("✍️ Abstractive Summarization")
        uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf", key="abstractive_uploader")
        
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            try:
                # Save temporary file
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("Generate Summary", key="abstractive_summary_btn"):
                    with st.spinner("Processing document..."):
                        # Extract text
                        st.info("📄 Extracting text from PDF...")
                        full_text = extract_text_from_pdf(temp_path)
                        
                        if not full_text:
                            st.error("❌ No text found in PDF. Please check the file.")
                        else:
                            # Split into chunks
                            st.info("🔁 Splitting into overlapping chunks...")
                            chunks = chunk_text_with_overlap(full_text, max_token_len=900, overlap=100)
                            st.info(f"✂️ Total Chunks: {len(chunks)}")
                            
                            # Generate summaries
                            st.info("🧠 Generating summaries...")
                            summaries = summarize_chunks(chunks)
                            
                            # Combine and display final summary
                            final_summary = "\n\n".join(summaries)
                            st.success("✅ Summary generated successfully!")
                            
                            # Display the summary in a nice format
                            st.subheader("📌 Final Abstractive Summary")
                            st.markdown("---")
                            st.markdown(final_summary)
                            st.markdown("---")
                            
                            # Optional: Add download button for the summary
                            st.download_button(
                                label="Download Summary",
                                data=final_summary,
                                file_name=f"summary_{uploaded_file.name}.txt",
                                mime="text/plain"
                            )
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    elif functionality == "Named Entity Recognition":
        st.header("🔍 Named Entity Recognition")
        uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf", key="ner_uploader")
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                case_text = extract_text_from_pdf(uploaded_file)
                st.info("Document uploaded successfully!")
                
                if st.button("Analyze Entities", key="analyze_entities_btn"):
                    with st.spinner("Analyzing entities..."):
                        legal_df, spacy_df = compare_ner_models(case_text)
                        st.success("Analysis complete!")

                        st.subheader("🟢 Legal NER Results")
                        st.dataframe(legal_df)  # or st.table(legal_df)

                        st.subheader("🔵 SpaCy NER Results")
                        st.dataframe(spacy_df)  # or st.table(spacy_df)


if __name__ == "__main__":
    main()




