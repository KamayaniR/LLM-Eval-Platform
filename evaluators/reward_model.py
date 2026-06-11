import os
import torch
from evaluators.base import BaseEvaluator
from typing import Optional

class RewardModelEvaluator(BaseEvaluator):
    def __init__(self, checkpoint_path: str = None):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path:
            self._load(checkpoint_path)

    def _load(self, checkpoint_path: str):
        print(f"Loading reward model from {checkpoint_path}...")
        from transformers import AutoTokenizer
        from reward_model.model import RewardModel

        if checkpoint_path.startswith("gs://"):
            local_path = "/tmp/reward_model_checkpoint.pt"
            from google.cloud import storage
            bucket_name = checkpoint_path.split("/")[2]
            blob_path = "/".join(checkpoint_path.split("/")[3:])
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            checkpoint_path = local_path

        self.model = RewardModel()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        print("Reward model loaded.")

    def score(self, prompt: str, response: str) -> Optional[float]:
        if self.model is None:
            return None
        try:
            text = prompt + " " + response
            inputs = self.tokenizer(
                text,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                reward = self.model(input_ids, attention_mask)

            raw = reward.item()
            normalized = torch.sigmoid(torch.tensor(raw)).item()
            return round(normalized, 4)

        except Exception as e:
            print(f"Reward model eval error: {e}")
            return None
