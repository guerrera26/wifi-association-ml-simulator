import pandas as pd
import matplotlib.pyplot as plt

# Load the full results
df = pd.read_csv("results_full.csv")


# 1. Average Per-Run Throughput (Baseline vs ML)

baseline_avg = df["Baseline Average Throughput"].mean()
ml_avg = df["ML Average Throughput"].mean()

plt.figure()
plt.bar(["Baseline", "ML"], [baseline_avg, ml_avg])
plt.title("Average Throughput Per Run (Baseline vs ML)")
plt.ylabel("Throughput (Mbps)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()



# 2. Average Sticky Events (Baseline vs ML)

baseline_sticky_mean = df["Baseline Sticky Events"].mean()
ml_sticky_mean = df["ML Sticky Events"].mean()

plt.figure()
plt.bar(["Baseline", "ML"], [baseline_sticky_mean, ml_sticky_mean], color=["blue", "orange"])
plt.title("Average Sticky Events Per Run (Baseline vs ML)")
plt.ylabel("Sticky Events")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()



# 3. Throughput Improvement Distribution
#    (ML throughput - Baseline throughput per run)

df["Throughput Improvement"] = df["ML Average Throughput"] - df["Baseline Average Throughput"]

plt.figure()
plt.hist(df["Throughput Improvement"], bins=30, edgecolor="black")
plt.title("Distribution of Throughput Improvement (ML - Baseline)")
plt.xlabel("Mbps Improvement")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

#Print summary stats to console
print("=== Summary Statistics ===")
print(f"Baseline Avg Throughput: {baseline_avg:.2f} Mbps")
print(f"ML Avg Throughput:       {ml_avg:.2f} Mbps")
print(f"Mean Improvement:         {df['Throughput Improvement'].mean():.2f} Mbps")
print(f"Baseline Sticky Events:   {baseline_sticky_mean:.2f}")
print(f"ML Sticky Events:         {ml_sticky_mean:.2f}")