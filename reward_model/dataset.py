from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

TOKENIZER_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256

class PreferenceDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        print(f"Loading HH-RLHF dataset ({split} split)...")
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        chosen = self.tokenizer(
            row["chosen"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        rejected = self.tokenizer(
            row["rejected"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(),
            "rejected_input_ids": rejected["input_ids"].squeeze(),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(),
        }
