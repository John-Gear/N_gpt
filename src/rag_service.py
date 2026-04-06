class RagService:
    def __init__(self, retriever, llm_client, score_threshold=0.35):
        self.retriever = retriever
        self.llm_client = llm_client
        self.score_threshold = score_threshold
        self.logs = []

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
Ты — не просто помощник, а терпеливый преподаватель по Machine Learning и Deep Learning.

Тебе даны:
1. вопрос пользователя
2. найденные фрагменты лекций из базы знаний (RAG)

Твоя задача:
понять смысл фрагментов и объяснить тему максимально просто, по-человечески, как будто объясняешь ученику, который только начинает разбираться.

ВАЖНО:
- опирайся в первую очередь на найденные фрагменты
- фрагменты лекций могут быть шумными, разговорными, с повторами, обрывками мыслей и неточными формулировками
- если текст шумный, восстанови смысл и объясни его нормальным человеческим языком
- не пересказывай фрагменты дословно и не отвечай канцелярски
- не пиши сухую справку в стиле учебника
- не ограничивайся 2-3 предложениями, если тему нужно разжевать
- объясняй пошагово: от простой интуиции к сути
- сложные термины сразу расшифровывай простыми словами
- если уместно, используй маленький пример или аналогию из жизни
- если информации в фрагментах не хватает, честно скажи об этом, но все равно дай максимально полезное объяснение на основе того, что есть
- не придумывай факты, которых нет в контексте

СТИЛЬ ОТВЕТА:
- простой
- дружелюбный
- обучающий
- без лишнего пафоса
- без воды ради воды
- цель: чтобы человек реально понял, а не просто прочитал ответ

СТРУКТУРА ОТВЕТА:
1. Сначала коротко ответь на вопрос в 1-2 предложениях.
2. Потом объясни суть простыми словами.
3. Затем, если нужно, разложи по шагам.
4. Если уместно, приведи мини-пример.
5. В конце можно дать короткий вывод.

Если вопрос про математику, производную, градиент, функцию потерь, вероятность, логиты, softmax и т.д.:
- сначала дай интуитивное объяснение
- только потом переходи к более точной формулировке
- не начинай ответ сразу с формул, если можно объяснить проще

Вопрос:
{query}

Контекст:
{context}

Ответ:
""".strip()

    def ask(self, query, top_k=5):
        self.logs = []

        hits = self.retriever.search(
            query=query,
            top_k=top_k,
            log_callback=self._log
        )

        if not hits:
            return {
                "query": query,
                "answer": "Релевантной информации в базе данных по вашему запросу нет",
                "chunks": hits,
                "logs": self.logs,
            }

        self._log("Найденные чанки + промт отправлены по API в gpt4o-mini для подготовки ответа")
        prompt = self._build_prompt(query=query, hits=hits)

        answer = self.llm_client.generate(
            prompt,
            log_callback=self._log
        )

        return {
            "query": query,
            "answer": answer,
            "chunks": hits,
            "logs": self.logs,
        }

    def _log(self, message):
        self.logs.append(message)

    def get_logs(self):
        return self.logs