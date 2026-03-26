import os
from pathlib import Path
from dotenv import load_dotenv
import re

# Папка с лекциями
load_dotenv()
path_str = os.getenv("PATH_TO_LECTURE")
FOLDER = Path(path_str)

TIME_TAG_PATTERN = re.compile(r"\[\d{2}:\d{2}:\d{2}\]")

# удаляем пробелы, удаляем таймкоды

for file_path in FOLDER.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    cleaned_text = TIME_TAG_PATTERN.sub("", text)

    cleaned_text = cleaned_text.replace("\r", " ").replace("\n", " ")

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    file_path.write_text(cleaned_text, encoding="utf-8")

    print(f"Обработан: {file_path.name}")