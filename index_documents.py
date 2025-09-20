"""
Indexing script for RAG pipeline.
This script loads a PDF or DOCX file, splits it into chunks using a fixed-size with overlap strategy,
generates embeddings using OpenAI embedding model, and stores them in a PostgreSQL database. 
The resulting chunks can later be used for retrieval-augmented generation (RAG).
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables (POSTGRES_URL and OPENAI_API_KEY from .env file).
# I have been told it's OK to use Open AI embedding model instead of google embedding model for this assignment.
load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL")) # Connect to Postgres

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
SPLIT_STRATEGY = "fixed-size with overlap"
CHUNK_SIZE = 500 # characters
CHUNK_OVERLAP = 50 # characters

def clean_chunk_text(text): 
    """
    Clean text before insertion into the database.
    Removes null characters and trims whitespace, since nulls can cause insertion failures in PostgreSQL.
    """
    return text.replace("\x00", "").strip()

def load_file(file_path):
    """
    Load a document from disk.
    Supports PDF and DOCX files. 
    Raises an error for unsupported formats.
    Args:
        file_path (str): Path to the input file.
    Returns:
        list: A list of LangChain Document objects.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

def split_text(docs):
    """
    Split documents into smaller chunks.
    Uses a recursive character splitter with a fixed chunk size of 500 characters and an overlap of 50 characters.
    Args:
        docs (list): List of LangChain Document objects.
    Returns:
        list: A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def insert_chunks_to_db(chunks, file_name):
    """
    Clean and embed chunks using OpenAI embeddings model.
    Then insert them with additional metadata into the PostgreSQL database.
    Args:
        chunks (list): List of LangChain Document chunks.
        file_name (str): The name of the input file.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    with engine.connect() as conn:
        for chunk in chunks:
            clean_chunk = clean_chunk_text(chunk.page_content)
            emb_vector = embeddings.embed_documents([clean_chunk])[0]
            query = text("""
                INSERT INTO chunks (chunk_text, embedding, filename, split_strategy, created_at)
                VALUES (:chunk_text, :embedding, :filename, :split_strategy, :created_at)
            """)
            # for each reacord a unique identifier ID will be automatically created.
            conn.execute(query, {
                "chunk_text": clean_chunk,
                "embedding": emb_vector,
                "filename": file_name,
                "split_strategy": SPLIT_STRATEGY,
                "created_at": datetime.now().astimezone()
            })
        conn.commit()


if __name__ == "__main__":
    file_path = input("Enter path to PDF or DOCX: ")
    docs = load_file(file_path)
    chunks = split_text(docs)
    insert_chunks_to_db(chunks, os.path.basename(file_path))
    print(f"Total chunks inserted: {len(chunks)}")
    print("Indexing complete!")