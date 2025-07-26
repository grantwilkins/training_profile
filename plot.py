import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("agg")  # Use a non-interactive backend for script execution
df = pd.read_csv("power-trace_2025-07-26-23-11-14.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
# sum rolling window of 8 samples
df["rolling_sum"] = df[" power.draw [W]"].rolling(window=8).sum()
plt.plot(df["timestamp"], df["rolling_sum"])
plt.xlabel("Time (s)")
plt.ylabel("Rolling Sum of Power Draw (W)")
plt.title("Rolling Sum of Power Draw Over Time")
plt.grid()
plt.savefig("rolling_sum_plot.png")  # Save the plot to a file
