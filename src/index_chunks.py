import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

chunked_file = os.getenv("PATH_TO_CHUNKED")
CHUNKS_FILE = Path(chunked_file)

COLLECTION_NAME = "lectures"

# модель embedding bge-m3
model = SentenceTransformer("BAAI/bge-m3")

# инициализируем векторную бд qdrant
vector_db = os.getenv("PATH_TO_VECTOR_DB")
client = QdrantClient(path=vector_db)

# создаем коллекцию
vector_size = model.get_sentence_embedding_dimension()

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

points = []

for i, chunk in enumerate(tqdm(load_chunks(CHUNKS_FILE))):
    text = chunk["text"]

    # embedding
    vector = model.encode(text).tolist()

    # payload это метаданные чанков
    payload = {
        "doc_id": chunk["doc_id"],
        "file_name": chunk["file_name"],
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk["chunk_index"],
        "text": text,
    }

    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload=payload
        )
    )

    # батч вставка каждые 100
    if len(points) >= 100:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        points = []

# остатки
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)

client.close()