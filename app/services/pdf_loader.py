"""
CLI entry-point for the PDF ingestion pipeline.

Usage
-----
    python -m app.services.pdf_loader ingest [--pdf-dir ./data/unstructured] [--force]
    python -m app.services.pdf_loader stats
    python -m app.services.pdf_loader query "macroeconomic outlook 2025"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    from app.ingestion.pipeline import ingest_folder, ingest_file

    if args.file_path:
        file_path = Path(args.file_path)
        stats = ingest_file(file_path=file_path, force=args.force)

        print("\n══════════════ Ingestion Summary ══════════════")
        print(f"  File processed : {file_path}")
        print(f"  Pages processed : {stats.pages_processed}")
        print(f"  Text chunks     : {stats.text_chunks}")
        print(f"  Table chunks    : {stats.table_chunks}")
        print(f"  Figure chunks   : {stats.figure_chunks}")
        print(f"  Chunks rejected : {stats.chunks_rejected}")
        print(f"  Total stored    : {stats.total_stored}")
        print(f"  Elapsed         : {stats.elapsed_seconds:.1f}s")
        print("═══════════════════════════════════════════════")
    else:
        pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
        stats = ingest_folder(pdf_dir=pdf_dir, force=args.force)

        print("\n══════════════ Ingestion Summary ══════════════")
        print(f"  Files processed : {stats.files_processed}")
        print(f"  Files skipped   : {stats.files_skipped}")
        print(f"  Pages processed : {stats.pages_processed}")
        print(f"  Text chunks     : {stats.text_chunks}")
        print(f"  Table chunks    : {stats.table_chunks}")
        print(f"  Figure chunks   : {stats.figure_chunks}")
        print(f"  Chunks rejected : {stats.chunks_rejected}")
        print(f"  Total stored    : {stats.total_stored}")
        print(f"  Elapsed         : {stats.elapsed_seconds:.1f}s")
        print("═══════════════════════════════════════════════")


def cmd_stats(args: argparse.Namespace) -> None:
    from app.ingestion.vectordb import IngestVectorStore

    store = IngestVectorStore()
    counts = store.collection_counts()
    print("\n══════════════ Vector Store Stats ══════════════")
    for name, count in counts.items():
        print(f"  {name}: {count} documents")
    print(f"  TOTAL: {store.total_count}")
    print("════════════════════════════════════════════════")


def cmd_query(args: argparse.Namespace) -> None:
    from app.ingestion.vectordb import IngestVectorStore

    store = IngestVectorStore()
    results = store.query(args.query, n_results=args.top_k)

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"── Result {i} (dist={r['distance']:.4f}, collection={r['collection']}) ──")
        print(f"   Source: {meta.get('source_file', '?')} p.{meta.get('page', '?')}")
        print(f"   Type: {meta.get('block_type', '?')} | Section: {meta.get('section', '')}")
        print(f"   Citation: {meta.get('citation', '')}")
        text = r["text"][:300]
        print(f"   Text: {text}{'…' if len(r['text']) > 300 else ''}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF Ingestion Pipeline CLI",
        prog="python -m app.services.pdf_loader",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest PDFs into vector store")
    p_ingest.add_argument("--file-path", type=str, default=None, help="Path to a single PDF file to ingest")
    p_ingest.add_argument("--pdf-dir", type=str, default=None, help="Override PDF directory")
    p_ingest.add_argument("--force", action="store_true", help="Re-ingest even if already processed")
    p_ingest.set_defaults(func=cmd_ingest)

    # stats
    p_stats = sub.add_parser("stats", help="Show vector store statistics")
    p_stats.set_defaults(func=cmd_stats)

    # query
    p_query = sub.add_parser("query", help="Query the vector store")
    p_query.add_argument("query", type=str, help="Search query")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
