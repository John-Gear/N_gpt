import os
import requests
from dotenv import load_dotenv

load_dotenv()

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

    def generate(self, prompt: str, log_callback=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.1,
            "max_tokens": 700,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()

        if log_callback:
            log_callback("LLM gpt4o-mini вернул ответ на Ваш вопрос")

        return data["choices"][0]["message"]["content"]