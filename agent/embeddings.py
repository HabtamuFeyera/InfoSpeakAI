import logging
import concurrent.futures
from typing import List, Dict, Any
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec

from agent.config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME

# Initialize the embedding function
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

def manage_pinecone_index() -> None:
    """Ensure the Pinecone index exists; create it if it does not."""
    try:
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME in existing_indexes:
            logging.info(f"Pinecone index '{INDEX_NAME}' already exists.")
        else:
            logging.info(f"Creating Pinecone index '{INDEX_NAME}'...")
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=spec)
            logging.info(f"Pinecone index '{INDEX_NAME}' created successfully.")
    except Exception as e:
        logging.exception(f"Error managing Pinecone index: {e}")

manage_pinecone_index()
index = pc.Index(INDEX_NAME)

def chunk_documents(df: pd.DataFrame, text_column: str, source: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split documents into manageable text chunks.
    """
    result = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for doc_idx, row in df.iterrows():
        text = row.get(text_column, "")
        if pd.isna(text) or len(text) < 100:
            continue  # Skip very short documents
        for i, chunk in enumerate(splitter.split_text(text)):
            result.append({
                "doc_id": f"{source}_{doc_idx}",
                "chunk_id": i,
                "text": chunk,
                "source": source
            })
    return result

def upsert_embeddings(df_chunks: pd.DataFrame) -> None:
    """
    Generate embeddings for document chunks and upsert them into the Pinecone index.
    """
    chunk_texts = df_chunks["text"].tolist()
    try:
        chunk_embeddings = embedding_function.embed_documents(chunk_texts)
    except Exception as e:
        logging.exception("Error generating embeddings.")
        return

    def process_batch(batch: List[tuple]) -> bool:
        try:
            index.upsert(vectors=batch)
            return True
        except Exception as e:
            logging.exception("Error upserting batch to Pinecone.")
            return False

    upsert_data = [
        (
            f"{row['doc_id']}_{row['chunk_id']}",
            chunk_embeddings[idx],
            {
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "source": row["source"]
            }
        )
        for idx, row in df_chunks.iterrows()
    ]
    batch_size = 50
    batches = [upsert_data[i:i + batch_size] for i in range(0, len(upsert_data), batch_size)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, batches))
    successful_batches = sum(results)
    logging.info(f"Upsert completed: {successful_batches} out of {len(batches)} batches succeeded.")
