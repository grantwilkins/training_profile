#!/usr/bin/env python3
"""
Plot GPU power consumption from the training trace
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("agg")  # Use a non-interactive backend for script execution


def parse_power_data(csv_file):
    """Parse the power trace CSV and aggregate power across 8 GPUs"""

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()

    # Parse power values (remove 'W' suffix and convert to float)
    df["power_watts"] = df["power.draw [W]"].str.replace(" W", "").astype(float)

    # Parse timestamps
    df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")

    # Group every 8 consecutive rows (representing 8 GPUs)
    df["group"] = df.index // 8

    # Aggregate by group: sum power, take first timestamp
    aggregated = (
        df.groupby("group")
        .agg(
            {
                "datetime": "first",
                "power_watts": "sum",
            }
        )
        .reset_index()
    )

    return aggregated


def plot_power_trace(df, output_file=None):
    """Plot total power consumption"""

    plt.figure(figsize=(12, 6))

    # Plot total power consumption
    plt.plot(df["datetime"], df["power_watts"], linewidth=1, color="red", alpha=0.8)
    plt.ylabel("Total Power (W)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.title(
        "GPU Power Consumption During Training (8×H100)", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    plt.show()


def main():
    csv_file = "power-trace_2025-07-26-23-11-14.csv"
    output_file = "power_trace_plot.png"

    print(f"Processing power data from: {csv_file}")

    # Parse and aggregate the data
    df = parse_power_data(csv_file)

    print(
        f"Processed {len(df)} time points (aggregated from {len(df) * 8} GPU readings)"
    )

    # Create the plot
    plot_power_trace(df, output_file)


if __name__ == "__main__":
    main()
