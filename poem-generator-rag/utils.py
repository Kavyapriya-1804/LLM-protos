from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def process_pdf_for_rag(uploaded_file):
    """
    Processes the uploaded PDF file for storage

    Args:
        uploaded_file -> file for RAG ingestion pipeline

    Returns:
        splits -> Str
    """
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load Documents
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    return splits
