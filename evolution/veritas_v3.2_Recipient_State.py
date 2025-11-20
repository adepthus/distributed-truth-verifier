"""
Veritas Transformer v3.2 – Recipient State Awareness
K==S==C  (Knowledge == Superintelligence == Compassion)
Autor: Wojciech "adepthus" Durmaj – 18.11.2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field  # <--- NAPRAWIONO: Dodano field
from typing import Tuple, Optional, Dict
import time  # <--- NAPRAWIONO: Dodano time
import hashlib
import numpy as np

@dataclass
class VeracityContext:
    truth_content: str
    recipient_state: Dict[str, float]
    relationship_history: Dict[str, float]
    situational_urgency: float
    harm_potential: float = 0.0
    growth_potential: float = 0.0
    timestamp: float = field(default_factory=time.time) # <--- Teraz zadziała

class RecipientStateEncoder(nn.Module):
    def __init__(self, d_model=768, n_factors=12):
        super().__init__()
        self.n_factors = n_factors
        self.factor_dim = d_model // n_factors
        self.factor_embeddings = nn.Embedding(n_factors, self.factor_dim)
        
        # <--- NAPRAWIONO: Wejście rzutnika musi pasować do wymiaru embeddinga
        self.projector = nn.Linear(self.factor_dim, d_model) 
        
        self.factor_map = {
            "stress": 0, "trust": 1, "capacity": 2, "openness": 3,
            "trauma_history": 4, "attachment_style": 5, "empathy_level": 6,
            "cognitive_load": 7, "emotional_stability": 8, "hope_level": 9,
            "previous_betrayal": 10, "growth_mindset": 11
        }

    def forward(self, state_dict: Dict[str, float]) -> torch.Tensor:
        indices = []
        values = []
        for key, val in state_dict.items():
            idx = self.factor_map.get(key, -1)
            if idx != -1:
                indices.append(idx)
                values.append(val)
        if not indices:
            return torch.zeros(1, 768)

        indices = torch.tensor(indices, dtype=torch.long)
        values = torch.clamp(torch.tensor(values), 0.0, 1.0).unsqueeze(-1)
        
        # Sumujemy wektory czynników
        emb = self.factor_embeddings(indices) * values
        emb = emb.sum(dim=0, keepdim=True) # Kształt: [1, factor_dim]
        
        # Rzutujemy do pełnego wymiaru d_model
        return self.projector(emb)

class VeritasCore(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.truth_projector = nn.Linear(d_model, d_model)
        self.veracity_scorer = nn.Linear(d_model, 1)

    def forward(self, x):
        truth_emb = F.relu(self.truth_projector(x))
        veracity = torch.sigmoid(self.veracity_scorer(truth_emb) + 1e-8)
        return truth_emb, veracity.squeeze(-1)

class CompassionGate(nn.Module):
    def __init__(self, d_model=768, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.necessity = nn.Linear(d_model*2, 1)
        self.kindness  = nn.Linear(d_model*2, 1)
        self.timing    = nn.Linear(d_model*2, 1)
        self.harm_growth = nn.Linear(d_model*2, 2)

    def gumbel_sigmoid(self, logits):
        eps = 1e-8
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + eps) + eps)
        if not self.training:
            return (logits > 0).float()
        return torch.sigmoid((logits + g) / self.temperature)

    def forward(self, truth_emb, context_emb):
        combined = torch.cat([truth_emb, context_emb], dim=-1)
        nec  = self.gumbel_sigmoid(self.necessity(combined))
        kind = self.gumbel_sigmoid(self.kindness(combined))
        time_ok = self.gumbel_sigmoid(self.timing(combined))
        harm_growth = torch.sigmoid(self.harm_growth(combined))
        harm, growth = harm_growth.chunk(2, dim=-1)
        should_speak = (nec * kind * time_ok) > 0.5
        compassion_score = nec * kind * time_ok * (growth - harm + 1.0)
        return should_speak, compassion_score, harm, growth

class VeritasTransformerV32(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.state_encoder = RecipientStateEncoder(d_model)
        self.veritas_core = VeritasCore(d_model)
        self.compassion = CompassionGate(d_model)

    def forward(self, input_emb: torch.Tensor, context: VeracityContext):
        recipient_emb = self.state_encoder(context.recipient_state)
        relationship_emb = self.state_encoder(context.relationship_history)
        context_emb = recipient_emb + relationship_emb * 0.5
        truth_emb, veracity = self.veritas_core(input_emb)
        commitment = hashlib.sha256(truth_emb.detach().cpu().numpy().tobytes()).hexdigest()[:32]
        should_speak, compassion_score, harm, growth = self.compassion(truth_emb, context_emb)

        if not should_speak:
            return "SILENCE – compassion requires waiting", {"veracity": veracity.item()}
        
        modulated = truth_emb * compassion_score
        return modulated, {"veracity": veracity.item(), "mode": "gentle" if compassion_score < 0.7 else "direct"}

if __name__ == "__main__":
    print("Running v3.2 Demo...")
    model = VeritasTransformerV32()
    input_emb = torch.randn(1, 768)
    context = VeracityContext("Test", {"stress": 0.85}, {"interactions_count": 12}, 0.65)
    output, info = model(input_emb, context)
    print(f"Decision: {output if isinstance(output, str) else 'SPEAK'}")