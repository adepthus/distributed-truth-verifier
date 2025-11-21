# -*- coding: utf-8 -*-
"""
Veritas Module: Swarm Consensus
Purpose: Sybil-resistant consensus mechanism based on Informational Energy.
Dependency: veritas_ockham
Author: Wojciech 'adepthus' Durmaj
"""

from dataclasses import dataclass
from typing import List
try:
    from veritas_ockham import OckhamGyroscope
except ImportError:
    # Fallback dla uruchomienia stand-alone
    from production.veritas_ockham import OckhamGyroscope

@dataclass
class NodeWitness:
    name: str
    narrative: str
    is_malicious: bool

class VeritasSwarm:
    def __init__(self):
        self.nodes: List[NodeWitness] = []
        self.engine = OckhamGyroscope()

    def add_node(self, name: str, narrative: str, is_malicious: bool = False):
        self.nodes.append(NodeWitness(name, narrative, is_malicious))

    def reach_consensus(self):
        print(f"\n--- VERITAS SWARM PROTOCOL (Nodes: {len(self.nodes)}) ---")
        
        claims = {}
        
        for node in self.nodes:
            # Waga głosu determinowana przez Ockham V2.1
            analysis = self.engine.analyze(node.narrative)
            weight = max(0.01, analysis.veritas_score)
            
            status = "⚠️ BOT " if node.is_malicious else "✅ NODE"
            print(f"{status} {node.name:12} | Score: {analysis.veritas_score:.4f} | Den: {analysis.semantic_density:.2f}")
            
            # Hashowanie treści dla grupowania
            narrative_hash = hash(node.narrative)
            if narrative_hash not in claims:
                claims[narrative_hash] = {
                    "text": node.narrative, 
                    "total_weight": 0, 
                    "voters": []
                }
            claims[narrative_hash]["total_weight"] += weight
            claims[narrative_hash]["voters"].append(node.name)

        print("\n--- CONSENSUS RESULT (Quality Weighted) ---")
        if not claims:
            print("No claims to process.")
            return

        winner = max(claims.values(), key=lambda x: x['total_weight'])
        
        for claim in claims.values():
            is_winner = claim == winner
            marker = "🏆 WINNER" if is_winner else "❌ REJECTED"
            
            print(f"{marker} | Strength: {claim['total_weight']:.4f} | Voters: {claim['voters']}")
            if is_winner:
                print(f"   -> ESTABLISHED TRUTH: \"{claim['text']}\"")

if __name__ == "__main__":
    # Demo
    swarm = VeritasSwarm()
    swarm.add_node("Alice", "Transakcja TXID:a1b2c3d4 potwierdzona w bloku 800000.", False)
    swarm.add_node("Bot_1", "Analiza heurystyczna sugeruje ewentualne opóźnienia.", True)
    swarm.add_node("Bot_2", "Weryfikacja proceduralna w ramach kontekstu sieci.", True)
    swarm.reach_consensus()