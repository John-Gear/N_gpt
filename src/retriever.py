import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()


class LectureRetriever:
    def __init__(
        self,
        collection_name: str,
        vector_db_path: str,
        model_name: str,
    ) -> None:
        self.collection_name = collection_name
        self.vector_db_path = vector_db_path
        self.model_name = model_name

        print("QDRANT PATH:", self.vector_db_path)
        print("COLLECTION:", self.collection_name)
        print("MODEL:", self.model_name)

        self.client = QdrantClient(path=self.vector_db_path)
        self.model = SentenceTransformer(self.model_name)

    def _embed_query(self, query: str) -> List[float]:
        query = query.strip()
        if not query:
            raise ValueError("Query is empty")

        formatted_query = (
            "Represent this sentence for searching relevant passages: " + query
        )

        vector = self.model.encode(
            formatted_query,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self._embed_query(query)

        must_conditions = []

        if doc_id:
            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            )

        if file_name:
            must_conditions.append(
                FieldCondition(
                    key="file_name",
                    match=MatchValue(value=file_name),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        hits = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                {
                    "score": point.score,
                    "doc_id": payload.get("doc_id"),
                    "file_name": payload.get("file_name"),
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "start_char": payload.get("start_char"),
                    "end_char": payload.get("end_char"),
                    "text": payload.get("text"),
                }
            )

        return hits

    def close(self) -> None:
        self.client.close()


def build_retriever_from_env() -> LectureRetriever:
    collection_name = os.getenv("QDRANT_COLLECTION")
    vector_db_path = os.getenv("PATH_TO_VECTOR_DB")
    model_name = os.getenv("EMBEDDING_MODEL")

    if not collection_name:
        raise ValueError("QDRANT_COLLECTION not found in environment")
    if not vector_db_path:
        raise ValueError("PATH_TO_VECTOR_DB not found in environment")
    if not model_name:
        raise ValueError("EMBEDDING_MODEL not found in environment")

    return LectureRetriever(
        collection_name=collection_name,
        vector_db_path=vector_db_path,
        model_name=model_name,
    )