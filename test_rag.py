from src.retriever import retriever
from src.llm_client import OpenRouterClient
from src.rag_service import RagService

def main():
    query = "расскажи про производную"

    llm_client = OpenRouterClient()
    rag_service = RagService(retriever=retriever, llm_client=llm_client)

    try:
        result = rag_service.ask(query=query, top_k=5)

        print("=" * 100)
        print("QUESTION:")
        print(result["query"])
        print("=" * 100)

        print("ANSWER:")
        print(result["answer"])
        print("=" * 100)

        print("RETRIEVED CHUNKS:")
        for i, chunk in enumerate(result["chunks"], start=1):
            print(f"\n[{i}] score={chunk['score']:.4f}")
            print(f"doc_id: {chunk.get('doc_id')}")
            print(f"chunk_id: {chunk.get('chunk_id')}")
            print((chunk.get("text") or "")[:500])
            print("-" * 100)

    finally:
        retriever.close()


if __name__ == "__main__":
    main()