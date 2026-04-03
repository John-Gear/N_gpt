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

qdrant_collections = os.getenv("QDRANT_COLLECTION")
COLLECTION_NAME = qdrant_collections

# модель embedding bge-m3
emb_model = os.getenv("EMBEDDING_MODEL")
model = SentenceTransformer(emb_model)

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
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"],
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