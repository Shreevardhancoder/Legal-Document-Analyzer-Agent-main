import torch
import argparse
import pdfplumber
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize

# Download nltk tokenizer if not already available
nltk.download("punkt")

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6").to(device)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text.strip()

# Overlapping chunking
def chunk_text_with_overlap(text, max_token_len=900, overlap=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    idx = 0

    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_len = len(sentence.split())

        if current_len + sentence_len <= max_token_len:
            current_chunk.append(sentence)
            current_len += sentence_len
            idx += 1
        else:
            chunks.append(" ".join(current_chunk))
            # overlap logic
            overlap_words = 0
            back_idx = len(current_chunk) - 1
            while back_idx >= 0 and overlap_words < overlap:
                overlap_words += len(current_chunk[back_idx].split())
                back_idx -= 1
            current_chunk = current_chunk[max(0, back_idx + 1):]
            current_len = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate summaries
def summarize_chunks(chunks, max_input_tokens=1024, max_output_tokens=150):
    summaries = []
    for i, chunk in enumerate(chunks):
        inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=max_input_tokens, truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=max_output_tokens, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"✅ Chunk {i+1}/{len(chunks)} summarized.")
        summaries.append(summary)
    return summaries

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abstractive Summarizer")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    print("📄 Extracting text from PDF...")
    full_text = extract_text_from_pdf(args.file)

    if not full_text:
        print("❌ No text found in PDF. Please check the file.")
        exit()

    print("🔁 Splitting into overlapping chunks...")
    chunks = chunk_text_with_overlap(full_text, max_token_len=900, overlap=100)
    print(f"✂️ Total Chunks: {len(chunks)}")

    print("🧠 Generating summaries...")
    summaries = summarize_chunks(chunks)

    final_summary = "\n\n".join(summaries)
    print("\n📌 Final Abstractive Summary:\n")
    print(final_summary)
