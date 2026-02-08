# Markdown

# 

# 

# \# HoloTSH: Synthetic Validation Experiment

# 

# This repository contains the simulation code for the paper:

# \*\*"HoloTSH: A Neuro-Symbolic Tensor Logic for TCM Modernization via Mathematical Isomorphism and Theoretical Guarantees"\*\*

# \*Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)\*.

# 

# \## 🧪 Experiment Description

# This code performs a Monte Carlo simulation (n=50 runs) to validate the \*\*HoloTSH Dual-Stream Architecture\*\*. It compares HoloTSH against standard HoRPCA under a \*\*70% missing data rate\*\* (simulating the "Data Wall" in TCM).

# 

# \### Key Metrics Verified:

# 1\. \*\*Mitigation of Shrinkage Bias:\*\* Validating Lemma 1 by measuring the recovery error of weak chronic pathological signals.

# 2\. \*\*Statistical Significance:\*\* T-test results comparing reconstruction errors.

# 

# \## 🚀 How to Run

# 

# 1\. \*\*Install Dependencies:\*\*

# &nbsp;  ```bash

# &nbsp;  pip install -r requirements.txt

# 

# 2.Run Simulation:

# Bash

# python holotsh\_final\_simulation.py

# 

# 📊 Results

# The simulation generates the following performance comparison:

# Metric	HoRPCA (Baseline)	HoloTSH (Ours)	Improvement

# Chronic Pathology RRE	~14.635	~1.594	~10x Reduction

# Anomaly Detection F1	~0.161	~0.163	Comparable

# Note: Results are generated dynamically and may vary slightly due to random seeds, though the seed is fixed to 2026 for reproducibility.

# 🔗 Citation

# If you use this code, please cite our IEEE JBHI paper (Citation details to be added upon publication).

