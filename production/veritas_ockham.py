# -*- coding: utf-8 -*-
"""
Veritas Module: Ockham's Gyroscope (V2.1)
Purpose: Epistemic energy measurement via Semantic Density vs Structural Entropy.
Author: Wojciech 'adepthus' Durmaj
"""

import zlib
import re
from dataclasses import dataclass

@dataclass
class NarrativeAnalysis:
    text: str
    structural_entropy: float
    semantic_density: float
    veritas_score: float

class OckhamGyroscope:
    def __init__(self):
        # Stop-words typowe dla "szumu biurokratycznego" i halucynacji LLM
        self.bureaucratic_stop_words = {
            "niniejszym", "zważywszy", "aspekt", "proceduralny", "weryfikacja", 
            "poziomie", "zakresie", "ramach", "kontekście", "ewentualne", 
            "szacunkowa", "heurystyczna", "estymacja", "analiza", "sugeruje",
            "hereby", "regarding", "framework", "context", "potential", "comprehensive"
        }

    def calculate_structural_entropy(self, text: str) -> float:
        """Mierzy 'gładkość' tekstu (kompresja)."""
        if not text: return 0.0
        data = text.encode('utf-8')
        return len(zlib.compress(data)) / len(data)

    def calculate_semantic_density(self, text: str) -> float:
        """Mierzy 'treściwość' tekstu (unikalne tokeny znaczące)."""
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        if not words: return 0.0
        
        unique_meaningful_words = {
            w for w in words 
            if w not in self.bureaucratic_stop_words and len(w) > 3
        }
        return len(unique_meaningful_words) / len(words)

    def analyze(self, text: str) -> NarrativeAnalysis:
        """Pełna analiza tekstu."""
        ent = self.calculate_structural_entropy(text)
        den = self.calculate_semantic_density(text)
        # Formuła Adepthusa: Promowanie gęstości (Sygnał), karanie Entropii (Koszt)
        score = (den * 3.0) - ent
        return NarrativeAnalysis(text, ent, den, score)