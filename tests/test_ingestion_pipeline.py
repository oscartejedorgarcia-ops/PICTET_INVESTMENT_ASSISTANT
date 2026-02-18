import os
import sys
import shutil

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from app.services.csv_loader import load_csv_to_sqlite
from app.ingestion.pipeline import ingest_folder

STORAGE_PATH = "storage"
CHROMADB_PATH = os.path.join(STORAGE_PATH, "chroma_db")
SQLITE_PATH = os.path.join(STORAGE_PATH, "sqlite")
RESOURCES_PATH = os.path.join(STORAGE_PATH, "resources")

def delete_storage():
    """Delete ChromaDB, SQLite databases, and resources."""
    if os.path.exists(CHROMADB_PATH):
        shutil.rmtree(CHROMADB_PATH)
        print(f"Deleted: {CHROMADB_PATH}")

    if os.path.exists(SQLITE_PATH):
        shutil.rmtree(SQLITE_PATH)
        print(f"Deleted: {SQLITE_PATH}")

    if os.path.exists(RESOURCES_PATH):
        shutil.rmtree(RESOURCES_PATH)
        print(f"Deleted: {RESOURCES_PATH}")

def ingest_data():
    """Ingest structured and unstructured data."""
    print("Ingesting structured data...")
    load_csv_to_sqlite()

    print("Ingesting unstructured data...")
    ingest_folder()

if __name__ == "__main__":
    delete_storage()
    ingest_data()