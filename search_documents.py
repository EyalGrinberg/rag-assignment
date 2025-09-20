"""
Search stored document chunks with Retrieval-Augmented Generation (RAG) and generate answers to user queries.

This script queries a PostgreSQL database containing text chunks and their embeddings, 
retrieves the most relevant chunks for a given user query, 
and uses an OpenAI LLM to generate an answer based on both the query and the retrieved context.

note: There is no real vector DB, just Postgres with an embedding column.
Thus, the RAG is implemented manually here.

Workflow:
1. Retrieve stored chunks and embeddings from the database (optionally filtered by filename).
2. Embed the user query using the same embedding model.
3. Compute cosine similarity between query and chunk embeddings.
4. Select the top-k most relevant chunks.
5. Build a RAG chain with the retrieved context.
6. Generate a final answer using an OpenAI LLM.
"""

from sqlalchemy import create_engine, text
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import numpy as np
import os

# Load environment variables (POSTGRES_URL and OPENAI_API_KEY)
load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL")) # Connect to Postgres

# Constants
LLM_MODEL = "gpt-4o-mini"  
EMBEDDING_MODEL = "text-embedding-3-large"

def fetch_chunks(filename=None):
    """
    Fetch text chunks and their embeddings from the database.
    Args:
        filename (str, optional): Filter results by source filename. If None, all chunks are retrieved.
    Returns:
        list[tuple]: A list of (chunk_text, embedding) tuples.
    """    
    with engine.connect() as conn:
        if filename:
            query = text("""
                SELECT chunk_text, embedding
                FROM chunks
                WHERE filename = :filename
            """)
            result = conn.execute(query, {"filename": filename})
        else:
            query = text("SELECT chunk_text, embedding FROM chunks")
            result = conn.execute(query)
        rows = result.fetchall()
    return rows

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(vec_a), np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_k_chunks(query, k=4, filename=None):
    """
    Retrieve the top-k most relevant chunks for a given query.
    Args:
        query (str): User query text.
        k (int, optional): Number of top chunks to return. Default is 4.
        filename (str, optional): Filter by source filename. Default is None.
    Returns:
        list[str]: List of top-k chunk texts ranked by similarity.
    """
    rows = fetch_chunks(filename)
    if not rows:
        return []
    # Embed the query
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    query_emb = embeddings.embed_query(query)
    # Compute similarity scores
    scored = []
    for chunk_text, chunk_emb in rows:
        score = cosine_similarity(query_emb, chunk_emb)
        scored.append((score, chunk_text))
    # Sort by score, descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk_text for _, chunk_text in scored[:k]]

def format_docs(docs):
    """
    Join retrieved chunks into a single context string.
    Args:
        docs (list[str]): List of chunk texts.
    Returns:
        str: Combined text with double newlines separating chunks.
    """
    return "\n\n".join(chunk_text for chunk_text in docs)

def build_rag_chain(top_docs):
    """
    Build a Retrieval-Augmented Generation (RAG) chain.
    Args:
        top_docs (list[str]): Top-ranked context chunks.
    Returns:
        Runnable: A LangChain pipeline that given a query and context returns an LLM-generated answer using the context.
    """
    prompt = ChatPromptTemplate([
        ("system", "You are a helpful assistant specialized in question answering. Use the provided context to answer the user query."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0) # Temperature=0 for deterministic output
    parser = StrOutputParser()
    # A more "modern" LangChain RAG chain with Runnables
    rag_chain = (
        {
            "question": RunnablePassthrough(), # Pass the user query as-is
            "context": RunnableLambda(lambda _: format_docs(top_docs)) # Format the retrieved docs (join them)
        }
        | prompt
        | llm
        | parser
    )
    return rag_chain


if __name__ == "__main__":
    query = input("Enter your question: ")
    k_input = input("Enter number of top documents (default is 4): ").strip()
    k = int(k_input) if k_input else 4
    filename = input("Enter filename (leave empty for all docs): ").strip() or None
    # Step 1: Retrieve top-k docs (to show user)
    top_docs = get_top_k_chunks(query, k, filename)
    print(f"\n--- Top {k} Retrieved Chunks ---")
    for d in top_docs:
        print(d)
        print("-" * 50)
    # Step 2: Generate answer with RAG
    chain = build_rag_chain(top_docs)
    answer = chain.invoke(query)
    print("\n--- Generated Answer ---")
    print(answer)