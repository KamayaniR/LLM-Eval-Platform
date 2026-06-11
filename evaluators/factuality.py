import os
import json
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

    def score(self, prompt: str, response: str, reference: str = None) -> Optional[float]:
        try:
            if reference:
                judge_prompt = f"""You are evaluating the factual accuracy of an AI response.

Question: {prompt}
Known correct answer: {reference}
AI response: {response}

How accurate is the AI response compared to the known correct answer?
1 = Completely wrong or contradicts the correct answer
2 = Mostly wrong with minor correct elements
3 = Partially correct but missing key facts
4 = Mostly correct with minor issues
5 = Fully accurate and consistent with the correct answer

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
            return (raw_score - 1) / 4.0

        except Exception as e:
            print(f"Factuality eval error: {e}")
            return None
