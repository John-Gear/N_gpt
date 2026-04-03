from typing import Dict, List, Any

class RagService:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "Контекст не найден."

        parts = []
        for i, hit in enumerate(hits, start=1):
            text = (hit.get("text") or "").strip()
            doc_id = hit.get("doc_id") or "unknown_doc"
            chunk_id = hit.get("chunk_id") or "unknown_chunk"
            score = hit.get("score")

            parts.append(
                f"[ФРАГМЕНТ {i}]\n"
                f"doc_id: {doc_id}\n"
                f"chunk_id: {chunk_id}\n"
                f"score: {score}\n"
                f"text:\n{text}"
            )

        return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(parts)

    def _build_prompt(self, query: str, hits: List[Dict[str, Any]]) -> str:
        context = self._build_context(hits)

        prompt = f"""
Ты помощник по лекциям Machine Learning и Deep Learning.

Тебе переданы:
1. Вопрос пользователя
2. Фрагменты лекций, найденные retriever'ом

Твоя задача:
- ответить на вопрос пользователя, опираясь ТОЛЬКО на переданные фрагменты
- если фрагменты частично шумные, вытащи из них полезный смысл и проигнорируй мусор
- если в контексте недостаточно информации для уверенного ответа, честно скажи об этом
- не придумывай факты, которых нет в контексте
- объясняй простым, понятным языком, как для новичка
- если это уместно, кратко упомяни, на каких фрагментах основан ответ

Вопрос пользователя:
{query}

Контекст из базы знаний (топ 5 чанков):
{context}

Сформируй итоговый ответ.
""".strip()

        return prompt

    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = self.retriever.search(query=query, top_k=top_k)
        prompt = self._build_prompt(query=query, hits=hits)
        answer = self.llm_client.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "chunks": hits,
            "prompt": prompt,
        }