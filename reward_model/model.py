import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

MODEL_NAME = "microsoft/deberta-v3-base"

class RewardModel(nn.Module):
    def __init__(self, pretrained_model_name=MODEL_NAME):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.reward_head = nn.Linear(self.config.hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        reward = self.reward_head(cls_output)
        return reward.squeeze(-1)
