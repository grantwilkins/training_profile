#!/usr/bin/env python3
"""
Plot GPU power consumption from the training trace
"""

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("agg")  # Use a non-interactive backend for script executionv


matplotlib.rcParams["pdf.fonttype"] = 42

matplotlib.rcParams["ps.fonttype"] = 42

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to write all text
        "font.family": "serif",
        "axes.labelsize": 14,  # LaTeX default is 10pt font.
        "font.size": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def parse_power_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df["power_watts"] = (
        df["power.draw [W]"].astype(float)
        if "power.draw [W]" in df.columns
        else df["power_watts"]
    )

    df["datetime"] = (
        pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
        if "timestamp" in df.columns
        else pd.to_datetime(df["datetime"])
    )

    df["group"] = df.index // 2
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
    start_time = aggregated["datetime"].iloc[0]
    aggregated["time_seconds"] = (
        aggregated["datetime"] - start_time
    ).dt.total_seconds()
    # rolling average of every 5 samples
    aggregated["power_watts"] = aggregated["power_watts"].rolling(window=1).mean()
    return aggregated


def plot_power_trace(df, output_file=None):
    plt.figure(figsize=(5, 3))
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(df["time_seconds"], df["power_watts"], color="red")
    plt.ylabel("Server Power (W)", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.xlim(0, df["time_seconds"].max())
    plt.ylim(0, 600)
    plt.grid(True)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    plt.show()


def main():
    csv_file = "../titanx-traces/power-trace_2gpu_2025-12-05-01-42-25.csv"
    output_file = f"../{csv_file.split('/')[-1].split('.')[0]}.pdf"

    print(f"Processing power data from: {csv_file}")

    # Parse and aggregate the data
    df = parse_power_data(csv_file)

    print(
        f"Processed {len(df)} time points (aggregated from {len(df) * 8} GPU readings)"
    )

    plot_power_trace(df, output_file)
    df.drop(columns=["group"], inplace=True)
    df.to_csv(
        f"../titanx-traces/{csv_file.split('/')[-1].split('.')[0]}_processed.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
