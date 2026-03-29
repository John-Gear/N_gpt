from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import os

# чистые лекции
INPUT_DIR = os.getenv("PATH_TO_LECTURE")

# обработанные чанки
OUTPUT_FILE = os.getenv("PATH_TO_CHUNKED")

CHUNK_SIZE = 1600

# нужен т.к. не жестко режем чанк а ищем точку "." < CHUNK_SIZE
MIN_CHUNK_SIZE = 1000

# Перекрытие между соседними чанками.
OVERLAP = 250

# Ф-ция для поиска "хорошей" точки в предложении (исключаем двоеточия, троеточия)
def is_valid_sentence_dot(text: str, dot_index: int) -> bool:
    if dot_index < 0 or dot_index >= len(text):
        return False

    if text[dot_index] != ".":
        return False

    # Проверка на троеточие
    prev_char = text[dot_index - 1] if dot_index - 1 >= 0 else ""
    next_char = text[dot_index + 1] if dot_index + 1 < len(text) else ""

    if prev_char == "." or next_char == ".":
        return False

    # Проверка на двоеточие
    if prev_char == ".." or next_char == "..":
        return False

    return True

# Ф-ция поиск конца чанка. 
def find_chunk_end(text: str, start: int, chunk_size: int, min_chunk_size: int) -> int:
    text_len = len(text)

    # Если вдруг старт уже за пределами текста — возвращаем конец текста
    if start >= text_len:
        return text_len

    # Целевой конец чанка
    target_end = min(start + chunk_size, text_len)

    # Минимальный размер чанка
    min_end = min(start + min_chunk_size, text_len)

    if target_end >= text_len:
        return text_len

    # Ищем точку от min_end до target_end включительно
    for i in range(target_end, min_end - 1, -1):
        if text[i] == "." and is_valid_sentence_dot(text, i):
            # чтобы точка вошла в чанк
            return i + 1

    # крайний случай режем по размеру чанка
    return target_end

# Ф-ция режет текстовую лекцию
def make_chunks(text: str, doc_id: str, file_name: str) -> List[Dict]:
    chunks: List[Dict] = []
    start = 0
    chunk_num = 1
    text_len = len(text)

    # цикл длинной на всю лекцию
    while start < text_len:
        end = find_chunk_end(
            text=text,
            start=start,
            chunk_size=CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
        )

        chunk_text = text[start:end].strip()

        # Защита от пустого чанка
        if not chunk_text:
            break

        chunks.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "chunk_id": f"{doc_id}_{chunk_num:04d}",
                "chunk_index": chunk_num - 1,
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
            }
        )

        # Если дошли до конца текста завершаем
        if end >= text_len:
            break

        # каждый следующий чанк учитывает overlap предидущего (чтобы конец предидущей мысли повторился в след. чанке)
        next_start = max(0, end - OVERLAP)

        # Защита от зацикливания (если вдруг overlap слишком большой и старт не сдвинулся)
        if next_start <= start:
            next_start = end

        start = next_start
        chunk_num += 1

    return chunks

# Ф-ция поочередной обработки файлов
def process_all_files(input_dir: Path) -> List[Dict]:
    all_chunks: List[Dict] = []

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"В папке {input_dir} не найдено .txt файлов")

    for file_path in txt_files:
        raw_text = file_path.read_text(encoding="utf-8")

        doc_id = file_path.stem
        file_name = file_path.name

        chunks = make_chunks(text=text, doc_id=doc_id, file_name=file_name)
        all_chunks.extend(chunks)

        print(f"{file_name}: {len(chunks)} chunks")

    # один общий список чанков по всем лекциям
    return all_chunks

# Ф-ция сохраняет чанки в формате JSONL
def save_jsonl(chunks: List[Dict], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

def main() -> None:
    chunks = process_all_files(INPUT_DIR)
    save_jsonl(chunks, OUTPUT_FILE)

    print(f"\nВсего чанков: {len(chunks)}")
    print(f"Сохранено в: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()