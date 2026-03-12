"""
run_experiments.py

Runs multiple simulation passes using:
    - The baseline RSSI-only policy
    - The ML-based association policy

For each pass:
    - Simulate one "world" with 5 clients
    - Record per-client throughput under both policies
    - Track sticky events if needed

Aggregates results across passes and writes them to 'results_full.csv'

This file is what you run when you want data for:
    - Tables 
    - Plots 
"""

import joblib
import pandas as pd
from Wifi_Association_Sim.simulator import simulate_run, baseline_policy, ml_policy


if __name__ == "__main__":
    # Load the previously trained ML model
    clf = joblib.load("model.joblib")

    rows = []  # holds per-run aggregated results

    # Number of simulated passes to run
    NUM_RUNS = 1000

    for _ in range(NUM_RUNS):
        # Run one pass with the baseline policy
        baseline_results, baseline_sticky = simulate_run(baseline_policy)

        # Run one pass with the ML policy
        ml_results, ml_sticky = simulate_run(ml_policy, clf=clf)

        # 5 clients per pass: compute average throughput
        baseline_avg = sum([entry[2] for entry in baseline_results]) / 5
        ml_avg = sum([entry[2] for entry in ml_results]) / 5

        # Append a single row per simulation run
        rows.append({
            "Baseline Average Throughput": baseline_avg,
            "ML Average Throughput": ml_avg,
            "Baseline Sticky Events": baseline_sticky,
            "ML Sticky Events": ml_sticky
        })

    # Convert records to a DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv("results_full.csv", index=False)
    print("Simulation completed. Results saved to results_full.csv")
