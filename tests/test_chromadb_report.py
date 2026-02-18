from chromadb import PersistentClient
from pathlib import Path
import warnings

# Suppress telemetry warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

def generate_chromadb_report():
    """Generate a report of the collections and the number of embeddings per collection in ChromaDB."""
    try:
        # Path to the ChromaDB directory
        chroma_dir = Path("storage/chroma_db")

        # Initialize ChromaDB client
        client = PersistentClient(path=str(chroma_dir))

        # Get all collection names
        collection_names = client.list_collections()

        report = []
        report.append(f"Number of collections: {len(collection_names)}\n")

        # Count embeddings in each collection
        for collection_name in collection_names:
            collection = client.get_collection(collection_name)
            count = collection.count()
            report.append(f"Collection: {collection_name}, Embeddings: {count}\n")

      

        return "".join(report)

    except Exception as e:
        return f"Error querying ChromaDB: {e}"

if __name__ == "__main__":
    report = generate_chromadb_report()
    print(report)