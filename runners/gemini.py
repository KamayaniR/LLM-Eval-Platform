import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class GeminiRunner:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("REGION", "us-central1")
        )

    def run(self, prompt: str, model: str = "gemini-2.5-flash", max_tokens: int = 512) -> str:
        try:
            result = self.client.models.generate_content(
                model=model,
                contents=prompt
            )
            return result.text.strip()
        except Exception as e:
            print(f"Gemini runner error: {e}")
            return ""
