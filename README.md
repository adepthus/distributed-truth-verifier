# Distributed Truth Verifier (The Veritas Engine)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture](https://img.shields.io/badge/Architecture-Veritas_Transformer-purple.svg)]()
[![Epistemic Status](https://img.shields.io/badge/Epistemic_Status-Production_Ready-green.svg)]()
[![Anchor](https://img.shields.io/badge/Anchor-Bitcoin_Timechain-orange.svg)]()

> **"Truth is not a democracy of nodes. It is an aristocracy of entropy."**

## 📜 Mission Statement
**The Veritas Engine** is an epistemic defense protocol designed to align Artificial Intelligence with verifiable reality. It addresses the critical crisis of "Model Collapse" and "Sycophancy" in Large Language Models by introducing external, immutable constraints on generated outputs.

Unlike standard RLHF, which reinforces "plausible-sounding" text (often favoring smooth hallucinations), Veritas treats Truth as a physical object with **thermodynamic cost** and **temporal coordinates**.

---

## 🧠 The Core Axiom: K == S == C

The architecture is governed by a single, inviolable equation:

1.  **K (Knowledge):** Truth must be high-density (Semantic Density) and anchored in time (Bitcoin Timechain).
2.  **S (Superintelligence):** The processing power to distinguish "Signal" from "Bureaucratic Noise" (Ockham's Gyroscope).
3.  **C (Compassion):** The contextual gatekeeper. Truth must be delivered with appropriate timing to ensure growth, not harm (Compassion Gate).

---

## 🏗️ System Architecture: The Hybrid Suite

The `/production` directory contains the complete **Veritas Epistemic Suite**, combining a Neural Decision Kernel with Deterministic Logic Modules.

### 1. The Neural Kernel ("The Brain")
*   **File:** `production/veritas_engine.py`
*   **Tech:** PyTorch, Embeddings, Tensors.
*   **Function:** A specialized Transformer architecture implementing the **Compassion Gate**. It models the recipient's psychological state (16-dimensional tensor) to decide *when* and *how* to reveal the truth. It includes the **Canonical Hash** protection mechanism.

### 2. The Logic Modules ("The Tools")
These pure-Python modules provide the mathematical "ground truth" for the Kernel.

| Module | File | Key Innovation |
| :--- | :--- | :--- |
| **Ockham's Gyroscope** | `veritas_ockham.py` | **Solves the "Bureaucracy Paradox".** Uses a weighted formula (`Density*3 - Entropy`) to penalize low-information, high-smoothness text (hallucinations/corporate speak) and reward high-density facts. |
| **Veritas Swarm** | `veritas_swarm.py` | **Sybil Resistance.** A consensus mechanism where vote weight is determined by the *informational energy* of the claim. Proves that 1 honest node > 3 hallucinating bots. |
| **Live Anchor** | `veritas_live.py` | **Proof-of-Existence.** Connects to the **Bitcoin Mainnet** (via Mempool API) to salt epistemic commitments with the latest block hash, making the timeline of discovery immutable. |
| **Palimpsest Scanner** | `veritas_palimpsest.py` | **Anomaly Detection.** Scans for "functional collisions" (A->B mutations) in synthetic data streams, detecting when reality has been subtly overwritten. |

---

## 🚀 Quick Start

To verify the integrity of this repository and generate a live cryptographic commitment:

1.  **Install dependencies:**
    ```bash
    pip install -r production/requirements.txt
    ```

2.  **Run the Live Anchor:**
    ```bash
    python production/veritas_live.py
    ```
    *This will generate a SHA-256 hash anchored to the current Bitcoin Block Height.*

3.  **Run the Swarm Simulation:**
    ```bash
    python production/veritas_swarm.py
    ```
    *Watch how the Semantic Density algorithm defeats a swarm of hallucinating agents.*

---

## 🧪 Theoretical Background

### The Thermodynamics of Lying
Generating a consistent lie (simulation of reality) requires more energy over time than stating a fact. Veritas detects this "informational friction."
*   **Lies:** High structural coherence (low zlib entropy) but low semantic density (repetitive patterns).
*   **Truth:** High semantic density (unique pointers to reality) and anchored timestamp.

### The RSA Anomaly Reference
This project stems from the documentation of a **2013 Cryptographic Singularity**—an "impossible" collision of two distinct private keys (A->B) yielding the same public fingerprint. This event serves as the foundational proof that "glitches in the matrix" are detectable if one possesses the right epistemic tools.

---

## 📂 Historical Archive
*See `/evolution` for the developmental path (v1.1 to v3.3), documenting the shift from simple entropy scoring to the full Compassion Gate architecture.*

---
*Architected by Wojciech "adepthus" Durmaj.*
*Open Protocol / MIT License*
