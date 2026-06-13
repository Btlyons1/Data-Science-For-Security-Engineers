# Data Science for Security Engineers

A comprehensive, hands-on guide designed for Detection Engineers to apply statistical modeling, behavioral analysis, and machine learning to security telemetry. This notebook provides a transition from simple threshold-based SIEM queries to robust, data-driven detections.

## Overview
Security data rarely follows a standard normal distribution. Traditional statistical methods (such as mean, standard deviation, and Pearson correlation) are sensitive to outliers and can produce misleading results when applied to heavy-tailed security telemetry, such as subnet byte transfers or high-cardinality user logs.

This repository contains the complete `Data_Science_for_Security_Engineers.ipynb` notebook. It is a **50-chapter curriculum** covering everything from foundational Exploratory Data Analysis (EDA) to advanced methods like Extreme Value Theory, Graph Topological Anomaly Detection (Oddball), and Deep Learning Autoencoders.

### Curriculum Structure
1. **Foundations (EDA & Metrics):** Exploratory Data Analysis, Robust Statistics (Median, MAD), Visualization Pitfalls.
2. **Distributions:** Normal vs Log-Normal vs Poisson, and how the distribution of your data determines which mathematical models are appropriate.
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

Open `Data_Science_for_Security_Engineers.ipynb` and run the cells sequentially. Note that some of the deep learning and matrix profile code blocks are computationally intensive and may take a few minutes to execute.

## The Detection Baseline Library (`detection_baseline`)

This repository also packages the core statistical, probabilistic, and behavioral detection engineering techniques from the guide into a clean, reusable local Python library. 

### Library Structure
- **`eda.py`**: Performs exploratory reconnaissance and decides on robust centering metrics (mean vs. median).
- **`statistics.py`**: Handles robust IQR/MAD calculations, modified Z-scores, adaptive EWMA baselines, CUSUM low-and-slow tracking, and Benford's Law distribution analysis.
- **`probabilistic.py`**: Implements Bayes' Theorem precision effectiveness, Fisher's Combined Probability test, and cost-benefit threshold optimization.
- **`behavioral.py`**: Measures transition anomaly scores (bigrams), Jaccard similarity, and Shannon entropy.
- **`automation.py`**: Analyzes timing regularity (Coefficient of Variation), session burstiness (Fano factor), and AI agent autonomy.
- **`detectors.py`**: Provides stateful first-seen entity detectors with contextual risk prioritization.
- **`datasets.py`**: Loads sample datasets (DGA domains, Benford-conforming network flows, and peer cohorts) for testing detection logic.

### Quick Start Example
To install the library locally in edit mode:
```bash
pip install -e .
```

You can then import and run robust security detections in your own scripts:
```python
from detection_baseline import datasets, detect_outliers_robust, optimize_detection_threshold

# 1. Load a sample network flow dataset
df_flows = datasets.load_network_flow_sample()

# 2. Detect outliers using robust Modified Z-Scores
results = detect_outliers_robust(df_flows['bytes_transferred'], threshold=3.5)
outliers = results[results['is_outlier']]
print(f"Detected {len(outliers)} anomalous network flow exfiltrations.")

# 3. Find the cost-optimal alert threshold (FP cost vs FN cost)
optimization = optimize_detection_threshold(
    scores=results['modified_zscore'].abs(),
    labels=(df_flows['node_id'] == 'node_attacker'),
    cost_fp=1.0,  # analyst triage cost
    cost_fn=15.0  # cost of missed breach
)
print(f"Optimal alert threshold: {optimization['optimal_threshold']:.2f}")
```


## Style & Philosophy
- **Grounded in Security Context:** Each algorithm is mapped directly to a concrete security use case, such as separating C2 beaconing from normal HTTP telemetry, or identifying horizontal scans using network graph metrics.
- **Focus on Robust Estimators:** The curriculum emphasizes robust statistics (like Median Absolute Deviation) and non-parametric distances rather than classical parametric assumptions, allowing you to model the extreme outliers native to security telemetry.
- **Practical Limitations:** Beyond code snippets, this guide explains the mathematical reasoning and discusses the failure modes and operational constraints of each detection in production.
