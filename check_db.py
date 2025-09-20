from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("POSTGRES_URL"))

filename_to_check = input("Enter the filename to check in DB (e.g., beagle.pdf): ").strip()

with engine.connect() as conn:
    result = conn.execute(
        text("""
            SELECT id, chunk_text, embedding, filename, split_strategy, created_at
            FROM chunks
            WHERE filename = :filename
            LIMIT 5
        """),
        {"filename": filename_to_check}
    )
    
    rows = result.fetchall()
    if not rows:
        print(f"No entries found for '{filename_to_check}'.")
    else:
        print(f"Showing up to 5 chunks for '{filename_to_check}':\n")
        for row in rows:
            emb_preview = row.embedding[:10] if row.embedding else None
            print(f"ID: {row.id}")
            print(f"Chunk text: {row.chunk_text[:100]}...")  # first 100 chars
            print(f"Embedding (first 10 dims): {emb_preview}")
            print(f"Filename: {row.filename}")
            print(f"Split strategy: {row.split_strategy}")
            print(f"Created at: {row.created_at}")
            print("-" * 50)