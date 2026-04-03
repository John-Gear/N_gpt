class RagService:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def ask(self, query: str, top_k: int = 5) -> dict:
        hits = self.retriever.search(query=query, top_k=top_k)
        prompt = self._build_prompt(query, hits)
        answer = self.llm_client.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "chunks": hits,
        }