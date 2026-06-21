"""Embed all chunks and store in ChromaDB."""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_DIR = PROJECT_ROOT / "output"
VECTORSTORE_DIR = PROJECT_ROOT / "output" / "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "hospitality_pms"
BATCH_SIZE = 256


def load_chunks() -> list[dict]:
    chunks = []
    for jsonl_file in [
        CHUNKS_DIR / "api_chunks" / "all_endpoints.jsonl",
        CHUNKS_DIR / "doc_chunks" / "all_doc_chunks.jsonl",
    ]:
        with open(jsonl_file) as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "module": c.get("module", ""),
            "doc_type": c.get("doc_type", ""),
            "source_file": c.get("source_file", c.get("spec_name", "")),
            "section": c.get("section", c.get("operation_id", "")),
            "method": c.get("method", ""),
            "path": c.get("path", ""),
        }
        for c in chunks
    ]

    # Deduplicate IDs (possible across api/doc chunks)
    seen = set()
    deduped = {"ids": [], "texts": [], "metadatas": []}
    for i, chunk_id in enumerate(ids):
        if chunk_id in seen:
            chunk_id = f"{chunk_id}_{i}"
        seen.add(chunk_id)
        deduped["ids"].append(chunk_id)
        deduped["texts"].append(texts[i])
        deduped["metadatas"].append(metadatas[i])

    print(f"Embedding {len(deduped['ids'])} chunks (batch size {BATCH_SIZE})...")
    for start in range(0, len(deduped["ids"]), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(deduped["ids"]))
        batch_texts = deduped["texts"][start:end]
        batch_ids = deduped["ids"][start:end]
        batch_meta = deduped["metadatas"][start:end]

        embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  {end}/{len(deduped['ids'])}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} vectors.")
    print(f"Stored at: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
