# Data Science for Security Engineers

A comprehensive, hands-on guide designed specifically for Detection Engineers to apply statistical thinking, behavioral modeling, and machine learning to security telemetry. This notebook bridges the gap between mathematically brittle SIEM queries and robust, data-driven Threat Intelligence.

## Overview
Security data is rarely "normal" in a textbook sense. Traditional statistics (Mean, Standard Deviation, Pearson Correlation) fail spectacularly when applied to violently heavy-tailed subnet bytes or high-cardinality user behaviors. 

This repository contains the complete `Data_Science_for_Security_Engineers.ipynb` notebook. It is a, **49-Chapter curriculum** guiding you from foundational Exploratory Data Analysis (EDA) all the way through Extreme Value Theory, Graph Topological Anomaly Detection (Oddball), and Deep Learning Autoencoders.

### Curriculum Structure
1. **Foundations (EDA & Metrics):** Exploratory Data Analysis, Robust Statistics (Median, MAD), Visualization Pitfalls.
2. **Distributions (The Engine):** Normal vs Log-Normal vs Poisson, and why the shape of your data legally dictates which mathematical models you can use.
3. **Outlier Detection:** Z-Scores, Log-Normal Transformations, Mahalanobis Distance.
4. **Frequencies & Probability:** Burst Analysis, HyperLogLog, Jaccard Similarity, Base Rate Problem (False Positive math), Pearson vs Spearman Correlation.
5. **Time-Series Analysis:** Stationarity, Seasonality, Beaconing Detection (Jitter & Modular Arithmetic).
6. **Advanced & Exotic Outliers:** 
    - Matrix Profiles (Time-Series Discards)
    - Local Outlier Factor (LOF)
    - Isolation Forests & DBSCAN 
    - One-Class SVM (Novelty Detection)
    - Autoencoders (Deep Learning)
    - Oddball (Graph-Based Ego-Network Topologies)
7. **The Horizon:** A theoretical look at the Bleeding Edge (Survival Analysis, SMOTE, Bayesian Causality, Reinforcement Learning, Conformal Prediction).

## Prerequisites

You will need Python 3.9+ and Jupyter installed. 
All required data science libraries are listed in `requirements.txt`.

### Installation

1. Clone this repository to your local machine.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK datasets (used for sequence analysis in Chapter 24) when running the notebook, or pull them automatically during execution.

## Usage

Start the Jupyter notebook server:
```bash
jupyter notebook
```

Open `Data_Science_for_Security_Engineers.ipynb` and begin running the cells sequentially. Ensure you have patience for some of the deep-learning and matrix profile code blocks, as they compute massive mathematical relationships instantly!

## Style & Philosophy
- **Geared for Detection Engineers:** Every single algorithm is immediately grounded in a concrete security application (e.g., separating Beaconing C2 from normal Slack telemetry, or finding a Horizontal Port Scan utilizing Graphic Node Densities).
- **Robust Algorithms over Classical:** We emphasize robust statistics (MAD) and nonparametric distances over classical parametric limits to handle the extreme outliers permanently natively to security events.
- **Deep Explanations:** Beyond surface-level code snippets, this guide provides the *intuition* behind the math, and explains exactly *why and where* these algorithms will fail in production environments.
