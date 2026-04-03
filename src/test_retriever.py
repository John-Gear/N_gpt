from retriever import build_retriever_from_env


def main() -> None:
    query = "что такое градиентный спуск"
    top_k = 5

    retriever = build_retriever_from_env()

    try:
        results = retriever.search(query=query, top_k=top_k)

        print("=" * 100)
        print(f"QUERY: {query}")
        print(f"TOP_K: {top_k}")
        print("=" * 100)

        if not results:
            print("Ничего не найдено")
            return

        for i, item in enumerate(results, start=1):
            print(f"\n[{i}] score={item['score']:.4f}")
            print(f"doc_id: {item['doc_id']}")
            print(f"file_name: {item['file_name']}")
            print(f"chunk_id: {item['chunk_id']}")
            print(f"chunk_index: {item['chunk_index']}")
            print(f"start_char: {item['start_char']}")
            print(f"end_char: {item['end_char']}")
            print("text:")
            print(item["text"])
            print("-" * 100)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()