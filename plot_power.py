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

    # Convert to seconds from start (t=0)
    start_time = aggregated["datetime"].iloc[0]
    aggregated["time_seconds"] = (
        aggregated["datetime"] - start_time
    ).dt.total_seconds()

    return aggregated


def plot_power_trace(df, output_file=None):
    """Plot total power consumption"""

    plt.figure(figsize=(5, 3))

    # Plot total power consumption
    plt.plot(df["time_seconds"], df["power_watts"])
    plt.ylabel("Server Power (W)", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.xlim(0, df["time_seconds"].max())
    plt.ylim(0, 5000)
    plt.grid(True)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    plt.show()


def main():
    csv_file = "power-trace_2025-07-26-23-32-09.csv"
    output_file = "power_trace_plot.pdf"

    print(f"Processing power data from: {csv_file}")

    # Parse and aggregate the data
    df = parse_power_data(csv_file)

    print(
        f"Processed {len(df)} time points (aggregated from {len(df) * 8} GPU readings)"
    )

    # Create the plot
    plot_power_trace(df, output_file)
    df.drop(columns=["group"], inplace=True)  # Clean up the DataFrame
    df.to_csv("aggregated_power_data.csv", index=False)


if __name__ == "__main__":
    main()
