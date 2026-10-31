"""One encoder, many heads: the separator and every council agent.

full pass  (context + user message):   tags (separator), intent, twist, purpose
plain pass (context + canonical text): decision (plain_request), field_class (severity),
                                       pattern (fraud_patterns), misuse (red_team), benign_need (advocate)
The plain pass never sees the story, so the claim-blind heads stay claim-blind.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from aegisframe.local.features import LABELS

POOLED = {"intent": len(LABELS["intent"]), "twist": len(LABELS["twist"]), "purpose": 1,
          "decision": len(LABELS["decision"]), "field_class": len(LABELS["field_class"]),
          "pattern": len(LABELS["pattern"]), "misuse": len(LABELS["misuse"]),
          "benign_need": len(LABELS["benign_need"])}


class AegisLocalModel(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.tags = nn.Linear(h, len(LABELS["tags"]))
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, n)) for k, n in POOLED.items()})

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kw).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # mean pooling
        pooled = self.dropout(pooled)
        out = {"tags": self.tags(self.dropout(hidden))}
        for k, head in self.heads.items():
            out[k] = head(pooled)
        return out


class ExportWrapper(nn.Module):
    """Flat outputs for ONNX: tags logits + one tensor per pooled head."""

    def __init__(self, model: AegisLocalModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        o = self.model(input_ids, attention_mask, token_type_ids)
        return tuple([o["tags"]] + [o[k] for k in POOLED])
