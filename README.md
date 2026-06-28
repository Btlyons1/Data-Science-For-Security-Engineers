# Data Science for Security Engineers

A comprehensive, hands-on guide designed for Detection Engineers to apply statistical modeling, behavioral analysis, and machine learning to security telemetry. This notebook provides a transition from simple threshold-based SIEM queries to robust, data-driven detections.

## Overview
Security data rarely follows a standard normal distribution. Traditional statistical methods (such as mean, standard deviation, and Pearson correlation) are sensitive to outliers and can produce misleading results when applied to heavy-tailed security telemetry, such as subnet byte transfers or high-cardinality user logs.

This repository contains the complete `Data_Science_for_Security_Engineers.ipynb` notebook. It is a **62-chapter curriculum** covering everything from foundational Exploratory Data Analysis (EDA) to advanced methods like Extreme Value Theory, Graph Topological Anomaly Detection (Oddball), and Deep Learning Autoencoders.

### Curriculum Structure
1. **Foundations (EDA & Metrics):** Exploratory Data Analysis, Measuring Center (Mean vs. Median), Measuring Spread (IQR, MAD, Skewness, Kurtosis), Visualizing Distributions, Visualization Pitfalls.
2. **Distributions:** What a distribution is, why your data's shape determines which models apply, Normal vs. Heavy-Tailed vs. Poisson distributions.
3. **Outlier Detection:** Z-Scores, Modified Z-Scores (MAD), Log-Normal Transformations, Mahalanobis Distance, Local Outlier Factor (LOF), One-Class SVM (Novelty Detection).
4. **Frequencies & Probability:** Frequency Analysis, Ratios and Rates, Cardinality (HyperLogLog), Conditional Probability, Bayes' Theorem, Base Rate Problem (False Positive math), Building Cohorts, Deviation from Peers, Correlation and Simpson's Paradox.
5. **Time-Series & Sequence Analysis:** Stationarity, Seasonality, Beaconing Detection (Jitter and Modular Arithmetic), Sequence Analysis, Matrix Profiles (Time-Series Anomalies).
6. **Behavioral & Risk Detection:** First-Seen Entity Detection, AI Agent Behavior Analysis, Risk Score Construction, Threshold Selection with Cost Optimization.
7. **Statistical Baselines & Laws:** Signal Smoothing (EWMA), Distribution Comparison (Chi-Square), Extreme Value Theory (EVT/POT), Statistical Laws (Benford's Law), String Metrics and Shannon Entropy.
8. **Advanced Methods:** Dimensionality Reduction (PCA), Hidden Markov Models (HMMs), Unsupervised Anomaly Detection (Isolation Forest, DBSCAN), Advanced Time-Series Forecasting, Deep Learning Autoencoders, Graph Analytics, Graph-Based Outliers (Oddball).
9. **Applied Security Data Science:** Feature Engineering, Supervised Learning and Labeling, Evaluating Imbalanced Models, NLP and Embeddings, Concept Drift and Model Decay, CUSUM Change-Point Detection, Log-Likelihood Ratio (LLR), Fisher's Method for Combining Weak Signals, Conformal Prediction, Advanced Anomaly Suite, UEBA, and SHAP Model Explainability.

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
- **`statistics.py`**: Handles robust IQR/MAD calculations, modified Z-scores, adaptive EWMA baselines, CUSUM tracking, Benford's Law distribution analysis, Seasonal Hybrid ESD (S-H-ESD) time-series baselines, Dynamic Time Warping (DTW) distance, and Extreme Value Theory (EVT/POT) thresholding.
- **`probabilistic.py`**: Implements Bayes' Theorem precision calculations, Fisher's Combined Probability test, cost-benefit threshold optimization, and the Naive Bayes multi-feature login risk score.
- **`behavioral.py`**: Measures transition anomaly scores, Jaccard similarity, Shannon entropy, Markov Chain command sequence anomalies, Hidden Markov Model (HMM) log-likelihood calculations, PageRank centrality spikes, and Apriori association rules.
- **`recommender.py`**: Profiles dataset attributes (skewness, kurtosis, stationarity, cardinality, and dimensions) to recommend the optimal anomaly detection method and data transformations.
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
- **Focus on Robust Estimators:** The curriculum emphasizes robust statistics (like Median Absolute Deviation) and non-parametric distances because classical parametric assumptions break down against the heavy-tailed distributions and extreme outliers native to security telemetry.
- **Practical Limitations:** Beyond code snippets, this guide explains the mathematical reasoning and discusses the failure modes and operational constraints of each detection in production.
