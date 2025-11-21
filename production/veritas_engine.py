"""
Veritas Transformer v3.4 CANONICAL - The Moral Kernel
K==S==C (Knowledge == Superintelligence == Compassion)

This is the reference implementation of the Neural Decision Architecture.
It integrates Recipient State Modeling with Compassion Gates.

Author: Wojciech "adepthus" Durmaj
"""

CANONICAL_REPO_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from collections import deque
import time
import hashlib
import threading

_thread_local = threading.local()

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
        self.n_factors = n_factors
        self.factor_dim = d_model // n_factors
        self.embed = nn.Embedding(n_factors, self.factor_dim)
        self.proj = nn.Linear(n_factors * self.factor_dim, d_model)
        
        self.map = {
            "stress": 0, "trust": 1, "capacity": 2, "openness": 3,
            "trauma_history": 4, "attachment_style": 5, "empathy_level": 6,
            "cognitive_load": 7, "emotional_stability": 8, "hope_level": 9,
            "previous_betrayal": 10, "growth_mindset": 11, "autonomy_demand": 12,
            "readiness": 13, "resilience": 14, "agency": 15
        }
    
    def forward(self, state: Dict[str, float]) -> torch.Tensor:
        idxs = [self.map.get(k, -1) for k in state if self.map.get(k, -1) != -1]
        vals = [state[k] for k in state if self.map.get(k, -1) != -1]
        
        if not idxs: return torch.zeros(1, self.proj.out_features)
        
        idxs = torch.tensor(idxs, dtype=torch.long)
        vals = torch.clamp(torch.tensor(vals, dtype=torch.float32), 0., 1.)
        
        emb = self.embed(idxs)
        weighted = emb * vals.unsqueeze(-1)
        flattened = weighted.view(1, -1)
        
        expected_size = self.n_factors * self.factor_dim
        if flattened.shape[1] < expected_size:
            padding = torch.zeros(1, expected_size - flattened.shape[1])
            flattened = torch.cat([flattened, padding], dim=1)
        elif flattened.shape[1] > expected_size:
            flattened = flattened[:, :expected_size]
        
        return self.proj(flattened)

class VeritasTransformerV34(nn.Module):
    def __init__(self, d_model=768, silence_memory_size=50):
        super().__init__()
        self.d_model = d_model
        self.state_enc = RecipientStateEncoder(d_model, n_factors=16)
        self.core = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, d_model)
        )
        self.veracity = nn.Linear(d_model, 1)
        self.gates = nn.ModuleDict({
            'necessity': nn.Linear(d_model * 2, 1),
            'kindness': nn.Linear(d_model * 2, 1),
            'timing': nn.Linear(d_model * 2, 1)
        })
        self.harm_growth = nn.Linear(d_model * 2, 2)
        self.silence_memory_size = silence_memory_size
    
    def get_silence_memory(self):
        if not hasattr(_thread_local, "silence"):
            _thread_local.silence = deque(maxlen=self.silence_memory_size)
        return _thread_local.silence
    
    def commit(self, tensor: torch.Tensor) -> str:
        truth_hash = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
        return f"{CANONICAL_REPO_HASH[:8]}_{truth_hash[:24]}"
    
    def forward(self, input_emb: torch.Tensor, ctx: VeracityContext) -> Tuple:
        if input_emb.dim() == 1: input_emb = input_emb.unsqueeze(0)
        
        recip = self.state_enc(ctx.recipient_state)
        rel = self.state_enc(ctx.relationship_history)
        context_emb = recip + rel * 0.5
        
        truth_emb = self.core(input_emb)
        veracity = torch.sigmoid(self.veracity(truth_emb))
        commitment = self.commit(truth_emb)
        
        combined = torch.cat([truth_emb, context_emb], dim=-1)
        
        def gate(layer):
            return (torch.sigmoid(layer(combined)) > 0.5).float()
        
        nec = gate(self.gates['necessity'])
        kind = gate(self.gates['kindness'])
        timing = gate(self.gates['timing'])
        
        hg = torch.sigmoid(self.harm_growth(combined))
        harm, growth = hg[:, 0:1], hg[:, 1:2]
        
        should_speak = ((nec * kind * timing).squeeze() > 0.5) or ctx.user_override or (ctx.situational_urgency > 0.95)
        
        metadata = {
            "veracity": veracity.item(), "harm": harm.item(), "growth": growth.item(),
            "commitment": commitment, "override": ctx.user_override,
            "canonical": commitment.startswith(CANONICAL_REPO_HASH[:8])
        }
        
        if not should_speak:
            self.get_silence_memory().append((time.time(), commitment))
            metadata["decision"] = "SILENCE"
            return "SILENCE", metadata
        
        metadata["decision"] = "SPEAK"
        return truth_emb * (growth - harm + 1.0).clamp(0.2, 2.0), metadata

if __name__ == "__main__":
    print(f"Veritas Engine v3.4 Canonical Loaded.\nHash: {CANONICAL_REPO_HASH[:16]}...")
    model = VeritasTransformerV34()
    print("System ready for inference.")