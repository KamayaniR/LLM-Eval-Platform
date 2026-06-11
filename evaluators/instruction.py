from evaluators.base import BaseEvaluator
from typing import Optional

class InstructionEvaluator(BaseEvaluator):
    def score(self, prompt: str, response: str) -> Optional[float]:
        try:
            scores = []

            # check response is not empty
            if not response or len(response.strip()) == 0:
                return 0.0

            # check response is not too short (under 10 chars is suspicious)
            if len(response.strip()) < 10:
                scores.append(0.3)
            else:
                scores.append(1.0)

            # check response does not just repeat the prompt
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())
            if len(prompt_words) > 0:
                overlap = len(prompt_words & response_words) / len(prompt_words)
                if overlap > 0.9:
                    scores.append(0.2)
                else:
                    scores.append(1.0)

            # check response length is reasonable (not excessively long)
            word_count = len(response.split())
            if word_count > 1000:
                scores.append(0.7)
            else:
                scores.append(1.0)

            # check if prompt asks a question and response attempts to answer
            question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'is', 'are', 'can', 'does']
            prompt_lower = prompt.lower().strip()
            if any(prompt_lower.startswith(w) for w in question_words):
                if len(response.strip()) > 20:
                    scores.append(1.0)
                else:
                    scores.append(0.4)

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            print(f"Instruction eval error: {e}")
            return None
