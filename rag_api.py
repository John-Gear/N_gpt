from flask import Flask, request, jsonify

from src.retriever import retriever
from src.llm_client import OpenRouterClient
from src.rag_service import RagService

app = Flask(__name__)

llm_client = OpenRouterClient()
rag = RagService(retriever=retriever, llm_client=llm_client)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}

    query = (data.get("query") or "").strip()
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be integer"}), 400

    try:
        result = rag.ask(query=query, top_k=top_k)

        return jsonify({
            "query": result["query"],
            "answer": result["answer"],
            "chunks": result["chunks"],
            "logs": result["logs"],
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/logger")
def logger():
    return jsonify({
        "logs": rag.get_logs()
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)