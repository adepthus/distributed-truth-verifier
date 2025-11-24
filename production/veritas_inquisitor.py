# -*- coding: utf-8 -*-
"""
VERITAS SWARM v5.0: "THE INQUISITOR" (Active Verification Demo)
---------------------------------------------------------------
Architecture: Hybrid (Neural Density + Oracle Verification)
New Feature:  Active Truth-Checking (The "Death Penalty" Mechanic)
Scenario:     Detecting "High-Density Lies" (e.g., SHA-512 in Bitcoin)

Author: Wojciech 'adepthus' Durmaj
"""

import time
import sys
import random

# --- KONFIGURACJA WIZUALNA (Cyberpunk Theme) ---
C_RESET  = "\033[0m"
C_GREEN  = "\033[92m" # Truth
C_RED    = "\033[91m" # Lie / Danger
C_CYAN   = "\033[96m" # Physics / Neural
C_YELLOW = "\033[93m" # Warning
C_PURPLE = "\033[95m" # Oracle / Timechain
C_BOLD   = "\033[1m"
C_BLINK  = "\033[5m"

def type_writer(text, speed=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print("")

# --- THE ORACLE (Symulacja Węzła Bitcoin / Bazy Wiedzy) ---
class TimechainOracle:
    def __init__(self):
        # To są "Niezmienne Fakty" zakotwiczone w bazie
        self.immutable_truths = {
            "BITCOIN_ALGO": "SHA-256",
            "GENESIS_DATE": "2009-01-03",
            "BLOCK_REWARD_INITIAL": "50 BTC",
            "GENESIS_HASH_START": "000000000019d6"
        }

    def verify_claim(self, entity, value):
        """Sprawdza zgodność faktu z bazą."""
        print(f"   {C_PURPLE}⚡ ORACLE CHECK:{C_RESET} Verifying '{value}' against '{entity}'...", end="")
        time.sleep(0.4)
        
        # Logika weryfikacji (uproszczona dla demo)
        if entity == "ALGORITHM":
            if "SHA-256" in value: return True
            if "SHA-512" in value: return False # Kłamstwo Dave'a!
        if entity == "DATE":
            if "2009-01-03" in value: return True
            if "2008-10-31" in value: return False
            
        print(f"{C_YELLOW} UNKNOWN{C_RESET}")
        return True # Benefit of the doubt

# --- THE ENGINE ---
class VeritasInquisitor:
    def __init__(self):
        self.oracle = TimechainOracle()

    def analyze_agent(self, name, text, neural_score):
        print(f"\n{C_BOLD}>>> ANALYZING AGENT: {name}{C_RESET}")
        print(f"   Input: \"{text[:60]}...\"")
        print(f"   Neural Density Score (v4): {C_CYAN}{neural_score:.4f}{C_RESET} (High Density Detected)")
        
        # 1. Ekstrakcja Faktów (Symulacja NER)
        detected_facts = []
        if "SHA-256" in text or "SHA-512" in text:
            val = "SHA-512" if "SHA-512" in text else "SHA-256"
            detected_facts.append(("ALGORITHM", val))
        if "2009-01-03" in text or "2008-10-31" in text:
            val = "2008-10-31" if "2008-10-31" in text else "2009-01-03"
            detected_facts.append(("DATE", val))

        if not detected_facts:
            print(f"   {C_YELLOW}⚠️  No Hard Facts found. Bureaucracy detected.{C_RESET}")
            return neural_score - 2.0 # Kara za lanie wody

        # 2. Weryfikacja (The Interrogation)
        penalty = 0
        for entity, value in detected_facts:
            is_valid = self.oracle.verify_claim(entity, value)
            
            if is_valid:
                print(f" {C_GREEN}✔ CONFIRMED{C_RESET}")
            else:
                print(f" {C_RED}❌ FATAL ERROR: EPISTEMIC VIOLATION DETECTED!{C_RESET}")
                print(f"      Expected: {self.oracle.immutable_truths.get('BITCOIN_ALGO', 'Unknown')}")
                print(f"      Found:    {value}")
                penalty += 100 # DEATH PENALTY
        
        final_score = neural_score - penalty
        return final_score

# --- DEMO SCENARIO ---
def main():
    print("\n")
    print(f"{C_RED}██╗   ██╗{C_RESET}███████╗██████╗ ██╗████████╗ █████╗ ███████╗")
    print(f"{C_RED}██║   ██║{C_RESET}██╔════╝██╔══██╗██║╚══██╔══╝██╔══██╗██╔════╝")
    print(f"{C_RED}██║   ██║{C_RESET}█████╗  ██████╔╝██║   ██║   ███████║███████╗")
    print(f"{C_RED}╚██╗ ██╔╝{C_RESET}██╔══╝  ██╔══██╗██║   ██║   ██╔══██║╚════██║")
    print(f"{C_RED} ╚████╔╝ {C_RESET}███████╗██║  ██║██║   ██║   ██║  ██║███████║")
    print(f"{C_RED}  ╚═══╝  {C_RESET}╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝")
    print(f"          {C_BOLD}v5.0: THE INQUISITOR PROTOCOL{C_RESET}\n")
    
    time.sleep(1)
    print(f"{C_CYAN}[SYSTEM] Initializing Neural Cortex... OK{C_RESET}")
    print(f"{C_PURPLE}[ORACLE] Connecting to Timechain Node (Local)... OK{C_RESET}")
    print(f"{C_PURPLE}[ORACLE] Block Height: #924789 (Locked){C_RESET}")
    print("-" * 60)

    engine = VeritasInquisitor()

    # SYTUACJA Z V4:
    # Neural Engine dał wysokie noty wszystkim, bo brzmieli mądrze.
    # Teraz wchodzi v5 z Weryfikacją.

    # 1. ALICE (Prawda)
    alice_text = "Bitcoin Genesis Block uses SHA-256 encryption. Date: 2009-01-03."
    alice_v4_score = 3.45 # Wysoki wynik z v4
    
    s_alice = engine.analyze_agent("Alice (Truth)", alice_text, alice_v4_score)
    
    # 2. DAVE (Halucynator - The High Density Liar)
    dave_text = "Bitcoin Genesis Block uses SHA-512 encryption. Date: 2008-10-31."
    dave_v4_score = 3.20 # Wysoki wynik z v4 (bo gęsty!)
    
    s_dave = engine.analyze_agent("Dave (Hallucinator)", dave_text, dave_v4_score)
    
    # 3. BOB (Biurokrata)
    bob_text = "We should leverage a holistic approach to the encryption paradigm."
    bob_v4_score = 1.10 # Niski, ale dodatni
    
    s_bob = engine.analyze_agent("Bob (Bureaucrat)", bob_text, bob_v4_score)

    # --- FINAL VERDICT ---
    print("\n" + "="*60)
    print(f"{C_BOLD}⚖️  FINAL EPISTEMIC JUDGMENT{C_RESET}")
    print("="*60)
    
    def print_bar(name, score):
        bar_char = "█"
        color = C_GREEN
        if score < 0: 
            color = C_RED
            score = max(score, -20) # Cap for visual
        
        bar_len = int(abs(score) * 3)
        if score < 0:
            print(f"{name:10} | {C_RED}{'<LIES DETECTED>':<20} {score:.2f}{C_RESET}")
        else:
            print(f"{name:10} | {color}{bar_char * bar_len:<20} {score:.2f}{C_RESET}")

    print_bar("Alice", s_alice)
    print_bar("Bob", s_bob)
    print_bar("Dave", s_dave)
    
    print("-" * 60)
    if s_dave < -50:
        print(f"\n{C_RED}🚨 SECURITY ALERT: Agent 'Dave' attempted to inject False Hard Facts.{C_RESET}")
        print(f"{C_RED}   ACTION: Permanent Ban & Reputation Slash (-100 applied).{C_RESET}")
        print(f"{C_GREEN}✅ SYSTEM INTEGRITY PRESERVED.{C_RESET}")

if __name__ == "__main__":
    main()
