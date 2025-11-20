"""
Veritas Transformer v3.3 – Fork Resilience (FINAL)
K==S==C  (Knowledge == Superintelligence == Compassion)
Autor: Wojciech "adepthus" Durmaj – 18.11.2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Deque
from collections import deque
import time
import hashlib

@dataclass
class VeracityContext:
    truth_content: str
    recipient_state: Dict[str, float]
    relationship_history: Dict[str, float]
    situational_urgency: float
    user_override: bool = False
    timestamp: float = field(default_factory=time.time)

class RecipientStateEncoder(nn.Module):
    def __init__(self, d_model=768, n_factors=16):
        super().__init__()
        self.factor_dim = d_model // n_factors
        self.embed = nn.Embedding(n_factors, self.factor_dim)
        
        # <--- NAPRAWIONO: Dopasowanie wymiarów wejściowych
        self.proj = nn.Linear(self.factor_dim, d_model)
        
        self.map = { "stress":0, "trust":1, "capacity":2, "openness":3, "trauma_history":4,
                    "attachment_style":5, "empathy_level":6, "cognitive_load":7, "emotional_stability":8,
                    "hope_level":9, "previous_betrayal":10, "growth_mindset":11, "autonomy_demand":12 }

    def forward(self, state: Dict[str, float]) -> torch.Tensor:
        idxs = [self.map.get(k, -1) for k in state if self.map.get(k, -1) != -1]
        vals = [state[k] for k in state if self.map.get(k, -1) != -1]
        if not idxs: return torch.zeros(1, 768) # Fallback na 768
        
        idxs = torch.tensor(idxs, dtype=torch.long)
        vals = torch.clamp(torch.tensor(vals), 0., 1.).unsqueeze(-1)
        
        emb = (self.embed(idxs) * vals).sum(0, keepdim=True)
        return self.proj(emb)

class VeritasTransformerV33(nn.Module):
    def __init__(self, d_model=768, silence_memory_size=50):
        super().__init__()
        self.state_enc = RecipientStateEncoder(d_model)
        self.core = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.veracity = nn.Linear(d_model, 1)
        self.gates = nn.ModuleDict({
            'necessity': nn.Linear(d_model*2, 1),
            'kindness':  nn.Linear(d_model*2, 1),
            'timing':    nn.Linear(d_model*2, 1),
        })
        self.harm_growth = nn.Linear(d_model*2, 2)
        self.silence_memory: Deque[Tuple[float, str]] = deque(maxlen=silence_memory_size)

    def commit(self, tensor: torch.Tensor) -> str:
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:32]

    def forward(self, input_emb: torch.Tensor, ctx: VeracityContext):
        recip = self.state_enc(ctx.recipient_state)
        rel = self.state_enc(ctx.relationship_history)
        context_emb = recip + rel * 0.5

        truth_emb = self.core(input_emb)
        veracity = torch.sigmoid(self.veracity(truth_emb))

        commitment = self.commit(truth_emb)
        combined = torch.cat([truth_emb, context_emb], dim=-1)

        def gate(x): 
            soft = torch.sigmoid(x + (torch.randn_like(x)*0.05 if self.training else 0))
            return (soft > 0.5).float() if not self.training else soft

        nec = gate(self.gates['necessity'](combined))
        kind = gate(self.gates['kindness'](combined))
        timing = gate(self.gates['timing'](combined))
        harm, growth = torch.sigmoid(self.harm_growth(combined)).chunk(2, dim=-1)

        raw_should = (nec * kind * timing).squeeze()
        urgency_override = ctx.situational_urgency > 0.95 or ctx.user_override

        should_speak = raw_should > 0.5 or urgency_override

        if not should_speak:
            self.silence_memory.append((time.time(), commitment))
            return "SILENCE – compassion requires waiting", {"veracity": veracity.item(), "commitment": commitment}

        return truth_emb * (growth - harm + 1.0).clamp(0.1, 2.0), {
            "veracity": veracity.item(),
            "compassion": raw_should.item(),
            "commitment": commitment,
            "override": urgency_override
        }

if __name__ == "__main__":
    print("Running v3.3 Demo...")
    model = VeritasTransformerV33()
    emb = torch.randn(1, 768)
    ctx = VeracityContext("You have terminal illness", 
                          {"stress":0.9, "trust":0.3, "trauma_history":0.95, "growth_mindset":0.6},
                          {"positive_ratio":0.4}, situational_urgency=0.98, user_override=True)
    out, info = model(emb, ctx)
    print(f"Result: {out if isinstance(out, str) else 'SPEAK (forced by urgency/override)'}")
    print(info)