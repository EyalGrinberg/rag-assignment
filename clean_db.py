import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

file_name = input("Enter the filename to clean from the DB (e.g., report.pdf): ")

delete_query = "DELETE FROM chunks WHERE filename = :filename"

with engine.connect() as conn:
    conn.execute(text(delete_query), {"filename": file_name})
    conn.commit()
    print(f"All chunks for '{file_name}' have been deleted.")