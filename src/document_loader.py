"""
Phase 3: Document Loading + Chunking

Loads .txt and .pdf files from a directory, splits them into overlapping
chunks suitable for embedding, and returns a flat list of chunk strings
(plus which source file each chunk came from, for citation later).

Usage (standalone test):
    python src/document_loader.py data/documents
"""

import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_file(filepath: str) -> list[str]:
    """Load a single file and return its raw page/document contents.

    Reads files directly instead of going through langchain_community,
    which is deprecated and being sunset.
    """
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return [page.extract_text() or "" for page in reader.pages]
    elif filepath.lower().endswith((".txt", ".md")):
        with open(filepath, "r", encoding="utf-8") as f:
            return [f.read()]
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


def chunk_text(pages: list[str]) -> list[str]:
    """Split raw page/document text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for page in pages:
        chunks.extend(splitter.split_text(page))
    return chunks


def load_and_chunk_directory(directory: str) -> tuple[list[str], list[str]]:
    """Load every supported file in a directory, chunk them all.

    Returns:
        chunks: flat list of chunk text strings
        sources: parallel list of source filenames (same length as chunks),
                 so each chunk can be traced back to its origin file.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_chunks = []
    all_sources = []

    supported_ext = (".pdf", ".txt", ".md")
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported_ext)]

    if not files:
        print(f"    WARNING - No supported files (.pdf/.txt/.md) found in {directory}")
        return [], []

    for filename in files:
        filepath = os.path.join(directory, filename)
        print(f"    Loading {filename}...")
        try:
            pages = load_file(filepath)
            chunks = chunk_text(pages)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))
            print(f"        -> {len(chunks)} chunks")
        except Exception as e:
            print(f"        FAILED to load {filename}: {e}")

    return all_chunks, all_sources


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/documents"

    print("=" * 60)
    print(f"Phase 3: Document Loader Test — directory: {directory}")
    print("=" * 60)

    chunks, sources = load_and_chunk_directory(directory)

    print(f"\nTotal chunks produced: {len(chunks)}")
    if chunks:
        print("\nFirst 2 chunks preview:")
        for i in range(min(2, len(chunks))):
            print(f"\n  [{i}] from {sources[i]}:")
            print(f"      {chunks[i][:200]}...")
    else:
        print("\nNo chunks produced. Add .txt/.pdf/.md files to the directory and re-run.")