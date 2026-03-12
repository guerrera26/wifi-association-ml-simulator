"""
train_model.py

Generates synthetic training data for the ML-based association policy
and trains a RandomForest classifier.

Each training sample describes one "world state":
    - RSSI to Main AP
    - RSSI to Ext AP
    - Backhaul capacity of Ext AP
    - Load on Main AP (number of connected clients)
    - Load on Ext AP

The label is:
    0 - Main AP gives higher throughput
    1 - Ext AP gives higher throughput

The trained model is saved to disk 'model.joblib'.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


def generate_data(n: int = 1000000):
    """
    Generate a random dataset of size n

    For each sample:
        - Randomly pick RSSIs for Main and Ext APs
        - Randomly pick Ext AP backhaul capacity
        - Randomly pick how many clients are already on each AP
        - Compute per-client throughput for Main and Ext under that state
        - Label the better AP as the "correct" choice

    Args:
        n (int): Number of samples to generate.

    Returns:
        X (np.ndarray): Feature matrix of shape (n, 5)
        y (np.ndarray): Label vector of shape (n,), values in {0, 1}
    """
    X = []
    y = []

    for _ in range(n): # (n = 1000000)
        # RSSI values: random integers between -80 dBm and -40 dBm
        rssi_main = np.random.randint(-80, -40)
        rssi_ext  = np.random.randint(-80, -40)

        # Backhaul capacity for Ext AP: between 40 and 180 Mbps
        backhaul = np.random.randint(40, 180)

        # Current load (number of clients) on each AP
        main_load = np.random.randint(0, 6)  # 0–5 clients
        ext_load  = np.random.randint(0, 6)

        # Throughput if client chooses Main AP
        thr_main = 300.0 / (main_load + 1)

        # Throughput if client chooses Ext AP
        thr_ext = min(300.0, backhaul) / (ext_load + 1)

        # Label: 0 if Main better, 1 if Ext better (tiebreaker favors Ext)
        best_ap = 0 if thr_main > thr_ext else 1

        # Features:
        # [rssi_main, rssi_ext, backhaul, main_load, ext_load]
        X.append([rssi_main, rssi_ext, backhaul, main_load, ext_load])
        y.append(best_ap)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # 1. Generate synthetic training data
    X, y = generate_data(n=1000000)

    # 2. Create and train a RandomForest classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # 3. Save the trained model to disk
    joblib.dump(clf, "model.joblib")
    print("Model trained and saved to model.joblib")