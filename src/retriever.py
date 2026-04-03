import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class LectureRetriever:
    def __init__(self, collection_name, vector_db_path, model_name):
        self.collection_name = collection_name
        self.client = QdrantClient(path=vector_db_path)
        self.model = SentenceTransformer(model_name)

    def _embed_query(self, query):
        query = "Represent this sentence for searching relevant passages: " + query.strip()
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def search(self, query, top_k=5):
        query_vector = self._embed_query(query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": point.score,
                **(point.payload or {})
            }
            for point in result.points
        ]

    def close(self):
        self.client.close()

load_dotenv()

collection_name = os.getenv("QDRANT_COLLECTION")
vector_db_path = os.getenv("PATH_TO_VECTOR_DB")
model_name = os.getenv("EMBEDDING_MODEL")

retriever = LectureRetriever(
    collection_name=collection_name,
    vector_db_path=vector_db_path,
    model_name=model_name,
)