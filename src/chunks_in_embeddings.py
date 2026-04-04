import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

CHUNKS_FILE = Path(os.getenv("PATH_TO_CHUNKED"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_DB_PATH = os.getenv("PATH_TO_VECTOR_DB")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# загружаем чанки
def load_chunks(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# переводим чанки в эмбединги (используем openrouter модель через апи)
def get_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]

def chunked(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found")

    client = QdrantClient(path=VECTOR_DB_PATH)

    all_chunks = list(load_chunks(CHUNKS_FILE))
    if not all_chunks:
        raise ValueError("No chunks found")

    # узнаём размер вектора на одном примере
    sample_embedding = get_embeddings([all_chunks[0]["text"]])[0]
    vector_size = len(sample_embedding)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    point_id = 0
    batch_size = 32

    for batch in tqdm(list(chunked(all_chunks, batch_size))):
        texts = [chunk["text"] for chunk in batch]
        embeddings = get_embeddings(texts)

        points = []
        for chunk, vector in zip(batch, embeddings):
            payload = {
                "doc_id": chunk["doc_id"],
                "file_name": chunk["file_name"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    client.close()
    print(f"Done. Indexed {point_id} chunks into collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()