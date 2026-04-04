class RagService:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

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
Ты отвечаешь по лекциям Machine Learning и Deep Learning.

Тебе переданы:
1. Вопрос пользователя
2. Фрагменты лекций, найденные retriever'ом

Твоя задача дать ответ пользователю. Соблюдай жесткие правила:
1. Отвечай только на основе переданных фрагментов.
2. Не добавляй знания из памяти, интернета или своих догадок.
3. Если информации во фрагментах недостаточно, так и скажи: "В найденных фрагментах недостаточно информации для полного ответа".
4. Не придумывай определения, которых нет в контексте.
5. Пиши просто и понятно, но без отсебятины.

Вопрос пользователя:
{query}

Контекст из базы знаний (топ 5 чанков):
{context}

""".strip()

    def ask(self, query, top_k=5):
        hits = self.retriever.search(query=query, top_k=top_k)
        prompt = self._build_prompt(query=query, hits=hits)
        answer = self.llm_client.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "chunks": hits,
        }