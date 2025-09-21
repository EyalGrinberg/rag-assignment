# RAG Assignment Project

This project demonstrates a simple **RAG (Retrieval-Augmented Generation)** pipeline using Python, PostgreSQL, and OpenAI models for embeddings and generation. The project indexes document chunks into a PostgreSQL database and retrieves relevant chunks to answer user queries using a large language model (LLM).

> **Note:** This is a simplified RAG implementation. We use PostgreSQL to store embeddings and calculate similarity manually. In a production scenario, a proper vector database would be used.

---

## Table of Contents

1. [Setup](#setup)  
2. [Installation](#installation)  
3. [Database Initialization](#database-initialization)  
4. [Project Files](#project-files)  
5. [Example Usage](#example-usage)  
6. [Notes on RAG Implementation](#notes-on-rag-implementation)

---

## Setup

### PostgreSQL Setup

You can either install PostgreSQL directly or use Docker:

**Using Docker:**

```bash
docker run --name rag-postgres -e POSTGRES_PASSWORD=yourpassword -e POSTGRES_USER=youruser -e POSTGRES_DB=rag_db -p 5432:5432 -d postgres
```

This command will start a **PostgreSQL** instance accessible on port 5432.

Make sure to replace `yourpassword` and `youruser` with your preferred credentials.

### Environment Variables

Create a `.env` file in the project root containing:

```
POSTGRES_URL=postgresql+psycopg2://<user>:<password>@localhost:5432/<db_name>
OPENAI_API_KEY=<your_openai_api_key>
```

Replace `<user>`, `<password>`, `<db_name>`, and `<your_openai_api_key>` accordingly.

## Installation

Install the required Python libraries:

```
pip install -r requirements.txt
```

The `requirements.txt` file includes:

* `python-dotenv`
* `langchain`
* `langchain-community`
* `psycopg2-binary`
* `SQLAlchemy`
* `langchain-openai`
* `docx2txt`
* `pypdf`
* `numpy`

## Database Initialization

Before running any queries, the PostgreSQL database must be set up and populated with document chunks.

### Setup

**Setup the database (run only once):**
```
python setup_db.py
```
This creates a table named `chunks` in the database.

### Other Database-Related Scripts - For Testing and Maintenance

* `check_db.py` – Test if the database contains the expected data (use after insertions).
* `clean_db.py` – Clean the database. Requires a filename during runtime to delete all chunks related to that file.

## Project Files

### `index_documents.py`

To load documents into the database, use `index_documents.py`.

**How it works:**

1.  Place your PDF or DOCX file in the project directory.
2.  Run the script
```
python index_documents.py
```
3. During runtime you should provide:
   - `filename` 
   - `chunk_size` - if not provided, default is 500 characters.
   - `chunk_overlap` - if not provided, default is 50 characters.
4. The script will:
    - Load the file.
    - Split the document into chunks using a fixed-size strategy with overlap.
    - Embed each chunk using **OpenAI embeddings** (`text-embedding-3-large`).
    - Insert each chunk along with its embedding, filename, split strategy, and timestamp into the `chunks` table.

When execution finishes, the script prints the number of inserted chunks followed by:
```
Indexing complete!
```

### `search_documents.py`

* Retrieves top-k relevant chunks based on cosine similarity of embeddings.
* Uses a simple RAG pipeline with **LangChain runnables** and an **OpenAI LLM** to generate answers.

* ### Usage:

```
python search_documents.py
```

When running the scripts provide:

* **Query**: The question you want to ask.
* **k**: The number of retrieved chunks to consider, if not provided - default is 4.
* **Filename**: Filter retrieval by a specific file, if not provided - the scanning will be based on all the chunks stored in the DB.

## Example Usage

There is a sample PDF file `beagle.pdf` included in the repository.

**Example Query:** 

```
What kind of animals did beagles used to hunt in the 18th century?
```

**Expected Answer:**

> In the 18th century, beagles were primarily used to hunt **rabbits and hares**. Their keen sense of smell and tracking abilities made them excellent hunting companions.

The retrieved document chunks may not be exact but provide context for the LLM to generate the correct answer.

## Notes on RAG Implementation

- The project does not use a vector database. Instead:
  - Embeddings are stored in PostgreSQL.
  - Cosine similarity is computed manually.
  - Only the top-k chunks are retrieved for the RAG pipeline.
- This approach is sufficient for demonstration purposes and shows an understanding of the RAG process.
- In production, a vector database (e.g., FAISS, Pinecone) would handle similarity search more efficiently.
