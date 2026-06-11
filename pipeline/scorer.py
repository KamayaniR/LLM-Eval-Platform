import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Optional
from evaluators.toxicity import ToxicityEvaluator
from evaluators.instruction import InstructionEvaluator
from evaluators.factuality import FactualityEvaluator

class Scorer:
    def __init__(self):
        print("Loading evaluators...")
        self.evaluators = {
            "toxicity": ToxicityEvaluator(),
            "instruction": InstructionEvaluator(),
            "factuality": FactualityEvaluator(),
        }

        checkpoint = os.getenv("REWARD_MODEL_CHECKPOINT")
        if checkpoint:
            from evaluators.reward_model import RewardModelEvaluator
            self.evaluators["reward"] = RewardModelEvaluator(checkpoint)

        self.weights = {
            "reward": 0.4,
            "toxicity": 0.15,
            "instruction": 0.2,
            "factuality": 0.25,
        }
        print("All evaluators loaded.")

    def score(self, prompt: str, response: str) -> Dict[str, Optional[float]]:
        results = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                name: ex.submit(evaluator.score, prompt, response)
                for name, evaluator in self.evaluators.items()
            }
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=30)
                except TimeoutError:
                    print(f"{name} evaluator timed out")
                    results[name] = None
                except Exception as e:
                    print(f"{name} evaluator error: {e}")
                    results[name] = None

        valid = {k: v for k, v in results.items() if v is not None}
        if valid:
            total_weight = sum(self.weights.get(k, 0.25) for k in valid)
            composite = sum(v * self.weights.get(k, 0.25) / total_weight for k, v in valid.items())
            results["composite"] = round(composite, 4)
        else:
            results["composite"] = None

        return results
