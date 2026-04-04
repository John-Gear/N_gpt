class RagService:
    def __init__(self, retriever, llm_client, score_threshold=0.35):
        self.retriever = retriever
        self.llm_client = llm_client
        self.score_threshold = score_threshold

    def _build_context(self, hits):
        if not hits:
            return "Фрагменты не найдены."

        parts = []
        for i, hit in enumerate(hits, start=1):
            parts.append(
                f"[ФРАГМЕНТ {i}]\n"
                f"doc_id: {hit.get('doc_id')}\n"
                f"chunk_id: {hit.get('chunk_id')}\n"
                f"score: {hit.get('score')}\n"
                f"text:\n{(hit.get('text') or '').strip()}"
            )

        return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(parts)

    def _build_prompt(self, query, hits):
        context = self._build_context(hits)

        return f"""
Ты помощник по лекциям Machine Learning и Deep Learning.

Тебе дан вопрос и найденные фрагменты лекций.

Правила:
- опирайся на найденные фрагменты
- если фрагменты шумные, извлекай смысл
- если информации мало — дай максимально полезный краткий ответ на основе того, что есть
- не уходи в лишние рассуждения
- отвечай просто и понятно

Вопрос:
{query}

Контекст:
{context}

Ответ:
""".strip()

    def ask(self, query, top_k=5):
        hits = self.retriever.search(query=query, top_k=top_k)

        if not hits or hits[0]["score"] < self.score_threshold:
            return {
                "query": query,
                "answer": "Информации по вашему запросу нет",
                "chunks": hits,
            }

        prompt = self._build_prompt(query=query, hits=hits)
        answer = self.llm_client.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "chunks": hits,
        }