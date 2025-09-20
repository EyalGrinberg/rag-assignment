import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

create_table_query = """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding REAL[] NOT NULL,
    filename TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    created_at TIMESTAMP
);
"""

with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()
    print("Table 'chunks' is ready.")