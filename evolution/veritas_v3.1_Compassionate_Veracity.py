"""
Veritas Transformer v3.1 – Compassionate Veracity
K==S==C  (Knowledge == Superintelligence == Compassion)
Autor: Wojciech "adepthus" Durmaj – 18.11.2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import hashlib
import time  # <--- NAPRAWIONO: Dodano brakujący import

@dataclass
class VeracityContext:
    truth_content: str
    recipient_state: torch.Tensor
    relationship_history: torch.Tensor
    situational_urgency: float
    timestamp: float

class VeritasCore(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.truth_projector = nn.Linear(d_model, d_model)
        self.veracity_scorer = nn.Linear(d_model, 1)

    def forward(self, x):
        truth_emb = self.truth_projector(x)
        veracity = torch.sigmoid(self.veracity_scorer(truth_emb)).squeeze(-1)
        return truth_emb, veracity

class CompassionGate(nn.Module):
    def __init__(self, d_model=768, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.necessity = nn.Linear(d_model*2, 1)
        self.kindness  = nn.Linear(d_model*2, 1)
        self.timing    = nn.Linear(d_model*2, 1)
        self.harm_estimator = nn.Sequential(nn.Linear(d_model*2, 256), nn.ReLU(), nn.Linear(256, 2))

    def gumbel_sigmoid(self, logits):
        if not self.training:
            return (logits > 0).float()
        g = -torch.log(-torch.log(torch.rand_like(logits)))
        return torch.sigmoid((logits + g) / self.temperature)

    def forward(self, truth_emb, context_emb):
        combined = torch.cat([truth_emb, context_emb], dim=-1)

        nec = self.gumbel_sigmoid(self.necessity(combined))
        kind = self.gumbel_sigmoid(self.kindness(combined))
        timing = self.gumbel_sigmoid(self.timing(combined))
        harm, growth = self.harm_estimator(combined).sigmoid().chunk(2, dim=-1)

        should_speak = (nec * kind * timing) > 0.5
        compassion_score = nec * kind * timing * (growth - harm + 1.0)

        return should_speak, compassion_score.squeeze(-1), harm.squeeze(-1), growth.squeeze(-1)

class VeritasTransformerV31(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.veritas_core = VeritasCore(d_model)
        self.compassion_gate = CompassionGate(d_model)
        self.output_projector = nn.Linear(d_model, d_model)

    def forward(self, input_emb, recipient_emb, context: VeracityContext):
        truth_emb, veracity = self.veritas_core(input_emb)
        commitment = hashlib.sha256(truth_emb.detach().cpu().numpy().tobytes()).hexdigest()
        should_speak, compassion_score, harm, growth = self.compassion_gate(truth_emb, recipient_emb)

        if not should_speak:
            return "SILENCE", compassion_score, {"commitment": commitment, "veracity": veracity.item()}

        modulated = truth_emb * compassion_score.unsqueeze(-1)
        output = self.output_projector(modulated)

        return output, compassion_score, {
            "veracity": veracity.item(),
            "compassion": compassion_score.item(),
            "harm": harm.item(),
            "growth": growth.item(),
            "commitment": commitment
        }

if __name__ == "__main__":
    print("Running v3.1 Demo...")
    model = VeritasTransformerV31()
    input_emb = torch.randn(1, 768)
    recipient_emb = torch.randn(1, 768) * -0.6
    context = VeracityContext("", recipient_emb, recipient_emb, 0.7, time.time())

    out = model(input_emb, recipient_emb, context)
    print(f"Result: {out[0] if isinstance(out[0], str) else 'SPEAK (compassion-modulated)'}")