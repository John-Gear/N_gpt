import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

class OpenRouterEmbeddingClient:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def embed(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

class LectureRetriever:
    def __init__(self, collection_name, vector_db_path, embedding_client):
        self.collection_name = collection_name
        self.client = QdrantClient(path=vector_db_path)
        self.embedding_client = embedding_client

    def _embed_query(self, query):
        query = query.strip()
        if not query:
            raise ValueError("Empty query")

        return self.embedding_client.embed([query])[0]

    def search(self, query, top_k=5, log_callback=None):
        # 1 этап
        if log_callback:
            log_callback("Ваш вопрос отправлен в по API в text-embedding-3-small для преобразования в embedding вектор")

        query_vector = self._embed_query(query)

        if log_callback:
            log_callback("Вопрос успешно преобразован в embedding-вектор")

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        hits = [
            {
                "score": point.score,
                **(point.payload or {}),
            }
            for point in result.points
        ]

        # 2 этап
        if log_callback:
            log_callback(f"По Вашему вопросу в базе данных найдено {len(hits)} релевантных чанков")

        return hits

    def close(self):
        self.client.close()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_DB_PATH = os.getenv("PATH_TO_VECTOR_DB")

embedding_client = OpenRouterEmbeddingClient(
    api_key=OPENROUTER_API_KEY,
    model=EMBEDDING_MODEL,
)

retriever = LectureRetriever(
    collection_name=COLLECTION_NAME,
    vector_db_path=VECTOR_DB_PATH,
    embedding_client=embedding_client,
)