from abc import ABC, abstractmethod
from typing import Optional

class BaseEvaluator(ABC):
    @abstractmethod
    def score(self, prompt: str, response: str) -> Optional[float]:
        pass
