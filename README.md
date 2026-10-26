# Legal Document Analysis System 📚⚖️

A comprehensive legal document analysis tool that provides multiple functionalities including extractive and abstractive summarization, named entity recognition, and legal Q&A capabilities.

## Features 🌟

### 1. Extractive Summarization 📝
- Extracts key points from legal documents
- Identifies related cases from the database
- Shows metadata including court information and document links
- Displays relevant points from similar cases

### 2. Abstractive Summarization ✍️
- Generates concise, human-readable summaries
- Handles long documents by splitting into chunks
- Maintains context through overlapping chunks
- Provides downloadable summary output

### 3. Named Entity Recognition 🔍
- Identifies legal entities using specialized legal NER model
- Compares results with general-purpose SpaCy NER
- Visualizes entity recognition results
- Supports multiple entity types

### 4. Legal Q&A System ❓
- Interactive chat interface for legal queries
- Uses InLegalBERT for semantic understanding
- Retrieves relevant context from legal database
- Powered by Gemini 2.5 Pro for accurate responses

## Installation 🛠️

1. Clone the repository:
```bash
git clone [repository-url]
cd legal-document-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Additional setup for Windows users:
```bash
pip install python-magic-bin
```

## Requirements 📋

```
streamlit
PyPDF2
torch
transformers
chromadb
nltk
numpy
pandas
google-generativeai
sentence-transformers
python-magic
python-magic-bin; platform_system == "Windows"
```

## Data Setup 📊

This repository contains only the codebase. To run the application, you'll need to:

1. **Download or prepare your legal dataset:**
   - Place legal documents in a `data/` directory
   - Ensure documents are in PDF format

2. **Initialize ChromaDB:**
   ```bash
   python setup_chroma.py
   ```

3. **Generate embeddings:**
   ```bash
   # Run the embedding notebook to populate ChromaDB
   jupyter notebook notebooks/embedding.ipynb
   ```

4. **Set up environment variables:**
   ```bash
   # Create .env file with your API keys
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

## File Structure (Code Only) 📁
```
legal-document-analyzer/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── setup_chroma.py          # ChromaDB initialization
├── summarization/           # Analysis modules
├── QnA_model/              # Q&A functionality
└── notebooks/              # Development notebooks
```

## Usage 🚀

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Choose functionality from the sidebar:
   - Extractive Summarization
   - Abstractive Summarization
   - Named Entity Recognition
   - Legal Q&A

4. Upload PDF documents or enter questions as needed

## Models Used 🤖

- **InLegalBERT**: Fine-tuned BERT model for legal domain
- **Gemini 2.5 Pro**: For generating natural language responses
- **Legal NER**: Specialized model for legal entity recognition
- **SpaCy Transformer**: For general-purpose NER comparison

## Database 🗄️

- Uses ChromaDB for vector storage and retrieval
- Maintains persistent storage of legal documents
- Enables semantic search capabilities
- Stores both document and sentence-level embeddings

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License 📄

[Your License Here]

## Acknowledgments 🙏

- Law-AI for InLegalBERT model
- Google for Gemini API
- ChromaDB team for vector database
- Streamlit for the web interface

## Contact 📧

[Your Contact Information]
