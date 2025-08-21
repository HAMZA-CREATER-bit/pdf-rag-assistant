
# PDF-Based Question Answering Bot (LangChain + DeepSeek + FAISS)

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about the contents of a PDF file. It uses:

- [LangChain](https://www.langchain.com/) for document processing and retrieval
- [HuggingFace Embeddings](https://huggingface.co/) for text embeddings
- [FAISS](https://faiss.ai/) for vector storage and retrieval
- [DeepSeek API](https://deepseek.com/) for generating responses

---

## Features
- Loads and splits PDFs into smaller text chunks for efficient retrieval.
- Uses HuggingFace embeddings to convert text into vector representations.
- Stores vectors in a FAISS database for fast similarity search.
- Integrates with DeepSeek's LLM for answering user queries based on the PDF content.
- Provides an interactive Q&A loop in the terminal.

---

## Project Structure
```
project/
│
├── main.py                # Main application code
├── .env                   # Environment variables (API key)
├── requirements.txt       # Required dependencies
└── README.md              # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
Create a `requirements.txt` with:
```txt
python-dotenv
langchain
langchain-community
langchain-huggingface
langchain-deepseek
faiss-cpu
```

Then install:
```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:
```ini
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

---

## Usage

1. Place your PDF in the project directory and update its path in `main.py`:
   ```python
   chunks = load_and_split_pdf("embedings.pdf")
   ```
2. Run the script:
   ```bash
   python main.py
   ```
3. Ask questions about the PDF:
   ```
   Ask a question (or type 'exit'): What is the main topic of this document?
   ```

---

## How It Works
1. **PDF Loading & Splitting** – Breaks the PDF into chunks for embedding.
2. **Embedding Generation** – Converts chunks into numerical vectors using HuggingFace models.
3. **Vector Storage (FAISS)** – Stores and retrieves the most relevant chunks.
4. **DeepSeek LLM** – Generates context-aware answers using retrieved chunks.
5. **Interactive Q&A Loop** – Allows users to ask multiple questions in real time.

---

## Future Improvements
- Add a web-based UI using Streamlit or Gradio.
- Support multiple PDFs and multi-document search.
- Enhance chunking strategy for better retrieval accuracy.
- Add caching for faster startup after first run.

---

## License
This project is open-source and available for personal and educational use.
