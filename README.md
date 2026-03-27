# Machine Learning Wi-Fi Association Simulator

Python simulation modeling wireless client association across multiple access points.  
This project evaluates whether a machine learning model can outperform traditional RSSI-based association policies in wireless networks.

---

# Overview

Wireless clients often remain connected to suboptimal access points even when a better one is available.  
This phenomenon is known as the **sticky client problem** and can reduce network efficiency, increase congestion, and reduce overall throughput.

This project simulates a wireless environment with:
- A Main Access Point
- An Extender Access Point
- Multiple simulated clients
- Network load and backhaul constraints
- Two association policies:
  - Traditional RSSI-based association
  - Machine learning-based association

The simulator evaluates whether a machine learning model can make better association decisions than signal-strength-only policies.

---

# Project Structure

```
Machine-Learning-WiFi-Simulator/
│
├── simulator.py
├── train_model.py
├── run_experiments.py
├── results_full.csv
├── model.joblib
└── README.md
```

## File Descriptions

### simulator.py
Core simulation engine.

Handles:
- Access Point objects
- Client objects
- Throughput calculations
- Sticky client detection
- Association policies
- Running simulation passes

### train_model.py
Generates synthetic training data and trains the machine learning model.

Outputs:
```
model.joblib
```

### run_experiments.py
Runs multiple simulation passes using:
- Baseline RSSI policy
- ML policy

Outputs:
```
results_full.csv
```

### Plotting / Analysis
Reads results and generates:
- Throughput comparison plots
- Sticky event comparison
- Throughput improvement histogram
- Summary statistics

---

# Simulation Model

## Network Model Assumptions

The simulator uses a simplified throughput model:

- Base Wi-Fi radio capacity = **300 Mbps**
- Extender AP may be limited by **backhaul capacity**
- Clients share bandwidth equally on an AP

### Throughput Model
```
Throughput per client =
min(Radio Capacity, Backhaul Capacity) / Number of Clients
```

This simplified model allows comparison between association policies.

---

# Machine Learning Model

## Features Used for Training

Each training sample represents one network state:

| Feature | Description |
|--------|-------------|
| RSSI Main | Signal strength to Main AP |
| RSSI Ext | Signal strength to Ext AP |
| Backhaul | Extender backhaul capacity |
| Main Load | Number of clients on Main AP |
| Ext Load | Number of clients on Ext AP |

## Labels
```
0 → Main AP gives higher throughput
1 → Ext AP gives higher throughput
```

## Model Used
```
RandomForestClassifier (scikit-learn)
```

The trained model is saved as:
```
model.joblib
```

---

# Technologies / Libraries Used

| Library | Purpose |
|--------|---------|
| Python | Programming language |
| NumPy | Random number generation and arrays |
| pandas | Data storage and CSV handling |
| scikit-learn | Machine learning model |
| joblib | Saving/loading trained model |
| matplotlib | Plotting results |

---

# Installation

## Install Dependencies
```
pip install -r requirements.txt
```

---

# How to Run the Project

## Step 1 — Train the ML Model
```
python train_model.py
```
This generates:
```
model.joblib
```

## Step 2 — Run Simulation Experiments
```
python run_experiments.py
```
This generates:
```
results_full.csv
```

## Step 3 — Plot Results
Run the plotting section to generate graphs and summary statistics.

---

# Output Files

| File | Description |
|------|-------------|
| model.joblib | Trained ML model |
| results_full.csv | Results from simulation runs |
| Plots | Throughput and sticky event comparisons |

---

# Future-Proofing / Maintenance Guide

This section explains how to maintain the project if libraries change or become deprecated.

## If scikit-learn Changes or Becomes Deprecated
You can replace the model with any classifier that supports:
```
fit(X, y)
predict(X)
```

Possible replacements:
- GradientBoostingClassifier
- DecisionTreeClassifier
- XGBoost
- LightGBM
- Neural Network (MLPClassifier)

Only the training file needs modification — the simulator will still work.

---

## If joblib Becomes Deprecated
Replace joblib with Python pickle.

### Save Model
```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### Load Model
```python
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## If matplotlib Becomes Deprecated
Replace plotting with:
- seaborn
- plotly
- pandas built-in plotting

The simulation results CSV will still work.

---

## If pandas Changes CSV Handling
Replace:
```
pd.read_csv()
df.to_csv()
```
With Python CSV module:
```python
import csv
```

---

# Reproducibility Notes

To make results reproducible, set random seeds:
```python
import numpy as np
np.random.seed(0)
```

---

# Possible Future Improvements

- Add more access points
- Add client mobility
- Use real RSSI datasets
- Reinforcement learning association policy
- Model interference
- Add network latency model
- Add visualization of network topology
- Hyperparameter tuning
- Docker container for reproducibility
- Configuration file for simulation parameters

---

# Summary

This project demonstrates that:
- RSSI alone is not sufficient for optimal client association
- Load and backhaul must be considered
- Machine learning can learn optimal association behavior
- ML policies can reduce sticky client problems

---

# Project Pipeline

```
train_model.py → model.joblib
run_experiments.py → results_full.csv
plot results → graphs
```
