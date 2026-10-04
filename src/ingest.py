"""Loads PDF documents into the Weaviate search index.

Run this once before asking questions (and again whenever you change
the files in the docs folder):

    python -m src.ingest --recreate

What it does, in order:
1. Reads every PDF in the docs folder and extracts its text.
2. Splits the text into overlapping chunks (see chunking.py).
3. Turns each chunk into an embedding, a list of numbers that captures
   its meaning, using a small model that runs on the local machine.
4. Stores every chunk plus its embedding in your Weaviate cluster,
   where both search methods (meaning and keyword) can find it.
"""

import argparse
from pathlib import Path

import weaviate
import weaviate.classes as wvc
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text
from src.config import settings
from src.weaviate_connect import connect


def read_pdf(path: Path) -> str:
    """Extracts the text from one PDF file."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ensure_collection(client: weaviate.WeaviateClient, recreate: bool):
    """Creates the collection (like a table) that will hold the chunks.

    vectorizer none means: we supply our own embeddings, computed
    locally, instead of letting the cloud compute them. Keyword (BM25)
    search needs no setup; Weaviate indexes the text for it automatically.
    """
    name = settings.collection_name
    if client.collections.exists(name):
        if recreate:
            client.collections.delete(name)
            print(f"Deleted the existing '{name}' collection, starting fresh.")
        else:
            print(
                f"Collection '{name}' already exists. "
                "Run with --recreate to replace it."
            )
            return client.collections.get(name)

    client.collections.create(
        name=name,
        vector_config=wvc.config.Configure.Vectors.self_provided(),
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(
                name="source",
                data_type=wvc.config.DataType.TEXT,
                # The file name is a label, not prose, so keep it out of
                # the keyword relevance scoring.
                index_searchable=False,
            ),
            wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
        ],
    )
    print(f"Created collection '{name}'.")
    return client.collections.get(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load PDFs into Weaviate.")
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Folder containing the PDF files (default: docs)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the collection before loading",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(
            f"No PDF files found in '{docs_dir}'. Put your PDFs there first."
        )

    print(f"Loading the embedding model ({settings.embedding_model_name})...")
    print("The first run downloads it (about 90 MB); later runs are instant.")
    embedder = SentenceTransformer(settings.embedding_model_name)

    client = connect()
    try:
        collection = ensure_collection(client, recreate=args.recreate)

        total = 0
        for pdf in pdf_files:
            text = read_pdf(pdf)
            chunks = chunk_text(text)
            if not chunks:
                print(f"  {pdf.name}: no text found, skipped.")
                continue

            # Encode all chunks of this file in one go (faster).
            vectors = embedder.encode(chunks, show_progress_bar=False)

            objects = [
                wvc.data.DataObject(
                    properties={
                        "text": chunk,
                        "source": pdf.name,
                        "chunk_index": i,
                    },
                    vector=vector.tolist(),
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            collection.data.insert_many(objects)
            total += len(objects)
            print(f"  {pdf.name}: {len(objects)} chunks loaded.")

        print(f"Done. {total} chunks from {len(pdf_files)} files are now searchable.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
