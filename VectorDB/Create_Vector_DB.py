import chromadb
from chromadb.utils import embedding_functions
import re
import os

def load_markdown(filepath):
    with open(filepath, "r") as f:
        return f.read()

def chunk_by_headers(markdown_text, min_level=2):
    """
    Splits markdown document into chunks based on headers.
    min_level=2 means split on ## and deeper headers.
    """
    pattern = r'(?=^#{' + str(min_level) + r',}\s)'
    chunks = re.split(pattern, markdown_text, flags=re.MULTILINE)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks

def extract_header(chunk):
    """Extracts the first header line as chunk ID/title."""
    lines = chunk.split("\n")
    for line in lines:
        if line.startswith("#"):
            return line.strip("#").strip()
    return "unknown"
def ingest_document(collection, filepath, doc_type, batch_size=500):
    markdown = load_markdown(filepath)
    chunks = chunk_by_headers(markdown)
    
    print(f"{filepath}: {len(chunks)} chunks found")
    
    if len(chunks) == 0:
        print(f"No chunks found, skipping")
        return

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        header = extract_header(chunk)
        chunk_id = f"{doc_type}_{i}_{header.replace(' ', '_').lower()}"

        documents.append(chunk)
        metadatas.append({
            "doc_type": doc_type,
            "source": os.path.basename(filepath),
            "header": header,
            "chunk_index": i
        })
        ids.append(chunk_id)

        # Flush batch when full
        if len(documents) >= batch_size:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            documents, metadatas, ids = [], [], []
            print(f"  Ingested batch up to chunk {i}")

    # Flush remaining
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    print(f"Ingested {len(chunks)} chunks from {filepath}")
def build_vector_db(
    persist_directory="credit_risk_db",
    documents={
        "Model_Explanations.md": "Model Explanations",
        "Logistic_Explanation.md": "Logistic Regression Interpretation",
        "Model_Performance_Metrics.md": "Logistic Model Performance Metrics",
        "allmodelmetrics.md": "Model Performance Metrics",
        "XGBoost_shapley_explain.md": "Shapley Value Interpretation",
        "Data_Explanation.md": "Data Interpretation",
        "historical_loan_data.md": "Loan Level Data",
        "logistic_configs_document.md": "Logistic Model Configs",
        "Shapley_Values.md": "Shapley Values Computed"
    }
):


    # Initialize client with persistence
    client = chromadb.PersistentClient(path=persist_directory)

    # Use sentence transformers for embeddings
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name="credit_risk",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Ingest each document
    for filepath, doc_type in documents.items():
        if os.path.exists(filepath):
            ingest_document(collection, filepath, doc_type)
        else:
            print(f"Warning: {filepath} not found, skipping")

    print(f"\nVector DB built successfully at {persist_directory}")
    print(f"Total chunks indexed: {collection.count()}")

    return collection


def query_collection(
    collection,
    query_text,
    n_results=5,
    doc_type_filter=None
):
  
    where = {"doc_type": doc_type_filter} if doc_type_filter else None

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where
    )

    return results


if __name__ == "__main__":
    collection = build_vector_db()

    # Example query
    results = query_collection(
        collection,
        query_text="What features most influence default probability?",
        n_results=3,
        doc_type_filter="shap"
    )

    for doc, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        print(f"\n--- {metadata['header']} ({metadata['doc_type']}) ---")
        print(doc[:300])