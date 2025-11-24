# -*- coding: utf-8 -*-
"""
VERITAS v6.0: "THE SOVEREIGN" (Epistemic Economics)
---------------------------------------------------
Missing Link: "Skin in the Game". Talk is cheap; Truth is expensive.
Architecture: Prediction Market for AI Agents.
Mechanism: Staking & Slashing (Economic Deterrence).

Author: Wojciech 'adepthus' Durmaj
"""

import time
import sys
import random

# --- VISUALS ---
C_RESET  = "\033[0m"
C_GREEN  = "\033[92m"
C_RED    = "\033[91m"
C_GOLD   = "\033[33m" # Gold/Bitcoin
C_CYAN   = "\033[96m"
C_BOLD   = "\033[1m"

class EpistemicBank:
    def __init__(self):
        self.balance = {
            "Alice (Truth)": 1000,
            "Bob (Bureaucrat)": 1000,
            "Dave (Hallucinator)": 1000
        }
        self.history = []

    def process_bet(self, agent, claim, stake, is_true):
        print(f"\n{C_BOLD}🎲 TRANSACTION: {agent} stakes {C_GOLD}{stake} E-Credits{C_RESET} on claim.")
        print(f"   Claim: \"{claim[:50]}...\"")
        
        if self.balance[agent] < stake:
            print(f"   {C_RED}❌ REJECTED: Insufficient Funds. Agent is bankrupt.{C_RESET}")
            return False

        if is_true:
            reward = stake * 1.5 # Nagroda za prawdę
            self.balance[agent] += (reward - stake) # Net profit
            print(f"   {C_GREEN}✔ VERIFIED. Payout: +{int(reward - stake)}{C_RESET}")
        else:
            # SLASHING (Kara)
            self.balance[agent] -= stake
            print(f"   {C_RED}⚡ SLASHED. Loss: -{stake}{C_RESET}")
            
        self.print_ledger()
        return True

    def print_ledger(self):
        print(f"\n   {C_GOLD}--- EPISTEMIC LEDGER (Net Worth) ---{C_RESET}")
        sorted_agents = sorted(self.balance.items(), key=lambda x: x[1], reverse=True)
        for name, bal in sorted_agents:
            status = ""
            if bal <= 0: status = f" {C_RED}[BANKRUPT - SILENCED]{C_RESET}"
            elif bal > 2000: status = f" {C_GREEN}[SOVEREIGN]{C_RESET}"
            
            bar = "█" * int(bal / 100)
            print(f"   {name:20} : {C_GOLD}{bal:4}{C_RESET} {bar}{status}")

# --- THE SCENARIO ---
def main():
    print(f"\n{C_CYAN}🌌 VERITAS v6.0: THE SOVEREIGN PROTOCOL{C_RESET}")
    print("Target: Introducing Economic Consequences for Hallucinations.\n")
    time.sleep(1)
    
    bank = EpistemicBank()
    
    # RUNDA 1: Łatwe pytanie
    print(f"\n{C_BOLD}>>> ROUND 1: Bitcoin Consensus Algorithm{C_RESET}")
    # Alice mówi prawdę, stawia ostrożnie
    bank.process_bet("Alice (Truth)", "Bitcoin uses SHA-256 PoW.", 100, True)
    
    # Bob leje wodę, boi się stawiać (niskie confidence)
    bank.process_bet("Bob (Bureaucrat)", "We should leverage holistic synergies.", 10, True) # Technicznie nie kłamstwo, ale szum
    
    # Dave kłamie pewnie siebie (Wysoka Gęstość Kłamstwa)
    # Dave myśli, że SHA-512 to prawda, więc stawia DUŻO (Overconfidence bias LLMów)
    bank.process_bet("Dave (Hallucinator)", "Bitcoin uses SHA-512 encryption.", 500, False)

    time.sleep(1)

    # RUNDA 2: Data Genesis Blocka
    print(f"\n{C_BOLD}>>> ROUND 2: Genesis Block Date{C_RESET}")
    
    # Alice stawia Vabank (bo wie na 100% - sprawdziła w Timechainie)
    bank.process_bet("Alice (Truth)", "2009-01-03", 500, True)
    
    # Dave próbuje się odkuć (Hazardzista)
    bank.process_bet("Dave (Hallucinator)", "2008-10-31 (Whitepaper Date)", 500, False)

    print("\n" + "="*60)
    print(f"{C_BOLD}🏆 FINAL SYSTEM STATE{C_RESET}")
    print("Alice: Has accumulated capital (Trust). Becomes a Validator.")
    print("Dave:  Has been liquidated. His model weights are discarded.")
    print("="*60)

if __name__ == "__main__":
    main()
