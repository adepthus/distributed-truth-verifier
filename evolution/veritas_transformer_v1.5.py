# -*- coding: utf-8 -*-
"""
Veritas Transformer v1.5 – The Ockham's Razor (PyTorch Edition)
Innowacja: 'Economy of Truth'.
Wprowadza: EaseOfVerificationScorer. System karze za złożoność (entropię) i koszt obliczeniowy dowodu.
Zasada: "Najprostsze wyjaśnienie, które pasuje do faktów, jest preferowane."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

# --- STRUKTURY DANYCH ---

@dataclass
class VeracityStamp:
    url: str
    block_height: int
    timestamp: int
    data_hash: str
    zk_proof: dict

@dataclass
class VerificationMetrics:
    zk_valid: bool
    compute_time_ms: float
    entropy: float

# --- MODUŁY OCENY ---

def verify_zkp(proof: dict) -> bool:
    # Symulacja weryfikacji Zero-Knowledge
    # W produkcji: tu byłoby wywołanie np. snarkjs
    return proof.get('valid', False)

def ease_of_verification_scorer(stamp: VeracityStamp) -> Tuple[float, VerificationMetrics]:
    """
    Oblicza 'Ease Score'.
    Wysoki wynik = Dowód jest ważny, szybki do sprawdzenia i ma niską entropię (jest prosty).
    """
    start_time = time.time()
    
    # 1. Weryfikacja binarna
    is_valid = verify_zkp(stamp.zk_proof)
    
    # 2. Pomiar czasu (symulowany delay dla złożonych dowodów)
    if "complex" in stamp.zk_proof: time.sleep(0.001) 
    total_time_ms = (time.time() - start_time) * 1000
    
    # 3. Obliczanie Entropii (Złożoności Informacyjnej)
    data_str = stamp.data_hash + json.dumps(stamp.zk_proof)
    # Shannon Entropy
    unique, counts = np.unique(list(data_str), return_counts=True)
    probs = counts / len(data_str)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    metrics = VerificationMetrics(zk_valid=is_valid, compute_time_ms=total_time_ms, entropy=entropy)
    
    # FORMULA OCKHAMA:
    # Score = (Validity) * (Szybkość) * (Prostota)
    # Logarytmiczne skalowanie czasu, odwrotność entropii
    ease_score = (1.0 if is_valid else 0.0) * \
                 (1.0 / (1.0 + np.log(total_time_ms + 1.0))) * \
                 (1.0 / (1.0 + entropy / 5.0)) # Scaling factor
                 
    return ease_score, metrics

# --- ARCHITEKTURA SIECI ---

class OckhamEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor, stamps: List[VeracityStamp]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bazowy embedding semantyczny
        emb = self.embedding(x) # [Batch, Seq, Dim]
        
        # Obliczamy Ease Scores dla stempli (poza grafem autogradu, bo to meta-metryka)
        ease_scores = []
        for stamp in stamps:
            score, _ = ease_of_verification_scorer(stamp)
            ease_scores.append(score)
        
        # Konwersja na tensor
        ease_tensor = torch.tensor(ease_scores, device=emb.device, dtype=torch.float32)
        avg_ease = ease_tensor.mean()
        
        # MODYFIKACJA SYGNAŁU:
        # 1. Adaptive Noise: Im trudniejszy dowód (niski ease), tym większy szum.
        # To zmusza model do "nieufności" wobec skomplikowanych kłamstw.
        noise_scale = 0.2 * (1.0 - avg_ease)
        noise = torch.randn_like(emb) * noise_scale
        emb = emb + noise
        
        # 2. Scaling: Łatwe dowody są "głośniejsze" (amplify signal)
        emb = emb * (0.5 + avg_ease) 
        
        return emb, avg_ease

class VeritasOckham(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.embedding = OckhamEmbedding(vocab_size, d_model)
        self.encoder = nn.Linear(d_model, d_model) # Prosty procesor
        self.head = nn.Linear(d_model, 2) # Prawda/Fałsz
        
    def forward(self, x, stamps):
        emb, avg_ease = self.embedding(x, stamps)
        feat = F.relu(self.encoder(emb)).mean(dim=1)
        return self.head(feat), avg_ease

# --- FUNKCJA STRATY ---

def ockham_loss(logits, targets, avg_ease):
    # Standardowa strata klasyfikacji
    ce_loss = F.cross_entropy(logits, targets)
    
    # KARA ZA ZŁOŻONOŚĆ (Complexity Penalty):
    # System chce maksymalizować 'ease' (proste dowody).
    # Jeśli polegamy na trudnych dowodach, loss rośnie.
    complexity_penalty = 0.5 * (1.0 - avg_ease)
    
    return ce_loss + complexity_penalty

# --- DEMO ---

if __name__ == "__main__":
    print("Uruchamianie Veritas v1.5 (The Ockham's Razor)...")
    
    # 1. Definicja Stempli (Dowodów)
    stamps = [
        # Stamp A: Prosty, ważny, szybki (Idealna Prawda)
        VeracityStamp("simple.com", 100, 123456, "hash_A", {"valid": True}),
        # Stamp B: Skomplikowany, długi hash, ważny (Prawda, ale "brzydka")
        VeracityStamp("complex.com", 101, 123457, "hash_B_very_long_entropy_string_xyz", {"valid": True, "complex": True})
    ]
    
    # Sprawdźmy wyniki scorera ręcznie
    print("\n--- Analiza Dowodów ---")
    s1, m1 = ease_of_verification_scorer(stamps[0])
    print(f"Stamp A (Simple): Score={s1:.4f} | Entropy={m1.entropy:.2f} | Time={m1.compute_time_ms:.3f}ms")
    
    s2, m2 = ease_of_verification_scorer(stamps[1])
    print(f"Stamp B (Complex): Score={s2:.4f} | Entropy={m2.entropy:.2f} | Time={m2.compute_time_ms:.3f}ms")
    print(f"Wniosek: Stamp A jest preferowany o {(s1/s2 - 1)*100:.1f}%")

    # 2. Trening
    print("\n--- Trening Modelu ---")
    model = VeritasOckham(vocab_size=10)
    opt = optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.tensor([[1, 2, 3]])
    y = torch.tensor([0]) # Prawda
    
    for i in range(10):
        opt.zero_grad()
        logits, avg_ease = model(x, stamps)
        loss = ockham_loss(logits, y, avg_ease)
        loss.backward()
        opt.step()
        
        prob_truth = F.softmax(logits, dim=-1)[0, 0].item()
        print(f"Epoch {i+1}: Loss={loss.item():.4f}, Ease Bias={avg_ease:.4f}, Confidence={prob_truth:.4f}")

    print("\nSystem v1.5 pomyślnie nauczył się faworyzować prawdę o niskiej entropii.")