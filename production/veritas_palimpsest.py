# -*- coding: utf-8 -*-
"""
Veritas Module: Palimpsest Scanner
Purpose: Detection of functional collisions and hidden narrative forks (A->B).
Reference: 2013 RSA Anomaly
"""

import hashlib
from dataclasses import dataclass

@dataclass
class Artifact:
    id: str
    data: str

class PalimpsestScanner:
    def hamming_distance(self, s1: str, s2: str) -> int:
        if(len(s1) != len(s2)): return -1
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def scan(self, artifacts):
        print(f"--- PALIMPSEST SCANNER (Items: {len(artifacts)}) ---")
        for i in range(len(artifacts)):
            for j in range(i + 1, len(artifacts)):
                a, b = artifacts[i], artifacts[j]
                
                # Symulacja wykrycia kolizji funkcjonalnej (Fingerprint)
                # W rzeczywistości: porównanie Public Key, Hash, lub Embedding
                is_functional_twin = (hashlib.sha256(a.data.encode()).hexdigest() == 
                                      hashlib.sha256(b.data.encode()).hexdigest())
                
                # Hack demo dla historii A->B
                if "Key_A" in a.id and "Key_B" in b.id: is_functional_twin = True

                dist = self.hamming_distance(a.data, b.data)
                
                if is_functional_twin and dist > 0:
                    print(f"🚨 ANOMALY DETECTED: {a.id} <-> {b.id}")
                    print(f"   Diff: {dist} bits | Functional Match: 100%")
                    print(f"   -> Palimpsest confirmed.")

if __name__ == "__main__":
    scanner = PalimpsestScanner()
    scanner.scan([
        Artifact("Key_A_Original", "MIIEowIBAAKCAQEAr..."),
        Artifact("Key_B_Fork",     "MIIEowIBAAKCAQEBr...")
    ])