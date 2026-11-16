"""Plot scenario results. Run after batching_model.py."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("output/scenario_results.csv")
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(df.scenario, df.avg_cost_per_order_inr, color="#4c78a8")
ax1.set_ylabel("Avg cost per order (INR)")
ax1.tick_params(axis="x", rotation=20)
ax2 = ax1.twinx()
ax2.plot(df.scenario, df.avg_delivery_time_min, color="#e45756", marker="o")
ax2.set_ylabel("Avg delivery time (min)")
plt.title("Batching rules: cost vs delivery time (synthetic data)")
plt.tight_layout()
plt.savefig("output/cost_vs_time.png", dpi=150)
print("Wrote output/cost_vs_time.png")
