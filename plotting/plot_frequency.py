import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_file_no_burn = "../titanx-traces/power-trace_2gpu_2025-12-04-01-22-31_processed.csv"
csv_file_with_burn = (
    "../titanx-traces/power-trace_2gpu_with_burn_2025-12-05-01-35-17_processed.csv"
)
output_file = "../titanx-traces/frequency_plot.pdf"


def compute_fft(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Parse timestamps and calculate time differences
    df["timestamp"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    df = df.sort_values("timestamp")
    time_diffs = df["timestamp"].diff().dt.total_seconds()
    avg_sample_rate = 1.0 / time_diffs.median()

    power_values = df["power_watts"].values

    # Calculate max ramp rate (per timestep)
    power_diffs = np.abs(np.diff(power_values))
    time_diffs_array = time_diffs.values[1:]  # Skip first NaN
    ramp_rates = power_diffs / time_diffs_array
    max_ramp_rate = np.max(ramp_rates)

    # Compute FFT
    fft_full = np.fft.fft(power_values)
    freqs_full = np.fft.fftfreq(len(power_values), d=1.0 / avg_sample_rate)
    positive_freqs = freqs_full[: len(freqs_full) // 2]
    positive_fft = np.abs(fft_full[: len(fft_full) // 2])

    # Normalize to fraction of total magnitude
    total_magnitude = np.sum(positive_fft)
    positive_fft_normalized = positive_fft / total_magnitude

    return positive_freqs, positive_fft_normalized, max_ramp_rate


# Compute FFT for both traces
freqs_no_burn, fft_no_burn, max_ramp_no_burn = compute_fft(csv_file_no_burn)
freqs_with_burn, fft_with_burn, max_ramp_with_burn = compute_fft(csv_file_with_burn)

# Print max ramp rates
print(f"Max ramp rate (without burn): {max_ramp_no_burn:.2f} W/s")
print(f"Max ramp rate (with burn): {max_ramp_with_burn:.2f} W/s")

# Plot both FFTs on the same plot
plt.figure(figsize=(7, 5))
plt.plot(freqs_no_burn, fft_no_burn, label="Without Burn", alpha=0.7)
plt.plot(freqs_with_burn, fft_with_burn, label="With Burn", alpha=0.7)
plt.xlabel("Frequency (Hz)")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Fraction of Total Magnitude")
plt.title("Power Trace FFT Comparison")
plt.legend()
plt.grid(True)
plt.savefig(output_file)
