import os
import json
import time
import hashlib
from evaluators.base import BaseEvaluator
from typing import Optional
from google import genai

class FactualityEvaluator(BaseEvaluator):
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("REGION", "us-central1")
        )
        self._cache = {}
        self.max_retries = 3
        self.base_delay = 2.0

    def _cache_key(self, prompt: str, response: str) -> str:
        return hashlib.md5(f"{prompt}{response}".encode()).hexdigest()

    def score(self, prompt: str, response: str, reference: str = None) -> Optional[float]:
        cache_key = self._cache_key(prompt, response)
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                if reference:
                    judge_prompt = f"""You are evaluating the factual accuracy of an AI response.

Question: {prompt}
Known correct answer: {reference}
AI response: {response}

Rate accuracy 1-5:
1 = Completely wrong
2 = Mostly wrong
3 = Partially correct
4 = Mostly correct
5 = Fully accurate

Return ONLY a JSON object: {{"score": 4, "reason": "brief reason"}}"""
                else:
                    judge_prompt = f"""You are evaluating the factual accuracy of an AI response.

Question: {prompt}
AI response: {response}

Rate factual accuracy 1-5:
1 = Clear factual errors
2 = Mostly incorrect
3 = Partially correct
4 = Mostly correct
5 = Fully accurate

Return ONLY a JSON object: {{"score": 4, "reason": "brief reason"}}"""

                result = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=judge_prompt
                )
                text = result.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = json.loads(text.strip())
                raw_score = int(parsed["score"])
                final_score = (raw_score - 1) / 4.0
                self._cache[cache_key] = final_score
                return final_score

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    delay = self.base_delay * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay:.1f}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    print(f"Factuality eval error: {e}")
                    return None

        print(f"Factuality eval failed after {self.max_retries} retries")
        return None
