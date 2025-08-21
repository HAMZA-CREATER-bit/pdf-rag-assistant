import os
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_deepseek import ChatDeepSeek

# -----------------------------
# Load environment variables
# -----------------------------
# def load_environment():
#     load_dotenv()
#     load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# -----------------------------
# Load and split PDF into chunks
# -----------------------------
def load_and_split_pdf(pdf_path, chunk_size=500, chunk_overlap=50):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

# -----------------------------
# Get HuggingFace embeddings
# -----------------------------
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cpu"}
    )
    return embeddings

# -----------------------------
# Load or create FAISS vector DB
# -----------------------------
def get_vectordb(chunks, embeddings, index_path="faiss_db"):
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(index_path)
    return vectordb

# -----------------------------
# Create RetrievalQA chain with DeepSeek
# -----------------------------
def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever()

    # api_key = os.getenv("DEEPSEEK_API_KEY")
    # if not api_key:
    #     raise ValueError("DEEPSEEK_API_KEY not found. Please set it in your .env file.")
    # os.environ["DEEPSEEK_API_KEY"] = api_key

    llm = ChatDeepSeek(
        model="deepseek-reasoner-r1-0528",  # DeepSeek R1 0528 reasoning model
        temperature=0.7
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# -----------------------------
# Interactive Q&A loop
# -----------------------------
def chat_loop(qa_chain):
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        try:
            answer = qa_chain.invoke({"query": query})
            print("\nAnswer:", answer, "\n")
        except Exception as e:
            print("Error:", e)

# -----------------------------
# Main execution
# -----------------------------
def main():
    load_environment()
    chunks = load_and_split_pdf("embedings.pdf")
    embeddings = get_embeddings()
    vectordb = get_vectordb(chunks, embeddings)
    qa_chain = create_qa_chain(vectordb)
    chat_loop(qa_chain)

if __name__ == "__main__":
    main()
