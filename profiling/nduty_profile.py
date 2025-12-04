#!/usr/bin/env python3
"""
nduty_calibrate.py – N×duty GEMM power calibration

Calibrates an empirical mapping (N, duty) -> avg GPU power (W) for a single GPU.

For each matrix size N and duty cycle in [0,1], we:
  * Allocate A, B ∈ R^{N×N} on the GPU.
  * Calibrate the average time per GEMM op.
  * Run a duty-cycled GEMM loop in fixed windows, sampling power via NVML.
  * Aggregate avg / stddev power and write to CSV.

Example:
    python nduty_calibrate.py \
        --gpu-id 0 \
        --Ns 1024,1536,2048,2560,3072,3584,4096 \
        --duties 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 \
        --window-s 0.2 \
        --total-duration-s 3.0 \
        --output nduty_calibration_gpu0.csv

You can then use the resulting CSV to build ramp schedules for startup/shutdown
and to choose (N, duty) pairs that match training power plateaus.
"""

import argparse
import csv
import math
import time
from typing import List

import torch

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


# ------------------------- NVML helpers -------------------------


def nvml_init():
    if not NVML_AVAILABLE:
        raise RuntimeError(
            "pynvml is not installed. Install with `pip install nvidia-ml-py3` "
            "or run your own external power logger."
        )
    pynvml.nvmlInit()


def nvml_get_handle(gpu_id: int):
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)


def read_power_W(handle) -> float:
    """Return instantaneous power in Watts from NVML."""
    # nvidia-smi power is in milliwatts
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def measure_idle_power(
    handle, duration_s: float = 5.0, interval_s: float = 0.2
) -> float:
    """Measure average idle power over duration_s."""
    readings = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        readings.append(read_power_W(handle))
        time.sleep(interval_s)
    return sum(readings) / max(1, len(readings))


# ------------------------- GEMM calibration -------------------------


def calibrate_op_time(
    a: torch.Tensor, b: torch.Tensor, gpu_id: int, num_ops: int = 10
) -> float:
    """
    Calibrate average time per GEMM (A @ B) in seconds for this N on this GPU.
    Uses CUDA events for accurate timing.
    """
    torch.cuda.synchronize(gpu_id)

    # Small warmup to avoid first-op overhead
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize(gpu_id)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_ops):
        _ = a @ b
    end.record()
    torch.cuda.synchronize(gpu_id)

    elapsed_ms = start.elapsed_time(end)
    op_time_s = (elapsed_ms / 1000.0) / num_ops
    return max(op_time_s, 1e-6)


# ------------------------- N×duty sweep core -------------------------


def run_nduty_sweep(
    gpu_id: int,
    Ns: List[int],
    duties: List[float],
    window_s: float,
    total_duration_s: float,
    idle_power_W: float,
    output_path: str,
):
    device = torch.device(f"cuda:{gpu_id}")
    handle = nvml_get_handle(gpu_id)

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "gpu_id",
            "N",
            "duty",
            "avg_power_W",
            "avg_power_minus_idle_W",
            "stddev_power_W",
            "num_samples",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in Ns:
            print(f"\n[INFO] GPU {gpu_id}: calibrating N={N}")
            # Allocate working buffers
            a = torch.randn(N, N, dtype=torch.float32, device=device)
            b = torch.randn(N, N, dtype=torch.float32, device=device)

            op_time_s = calibrate_op_time(a, b, gpu_id)
            print(f"[INFO]   op_time_s ≈ {op_time_s * 1e3:.3f} ms")

            for duty in duties:
                if duty < 0.0 or duty > 1.0:
                    continue

                print(f"  [SWEEP] duty={duty:.4f} ...", end="", flush=True)

                powers = []
                t_end = time.time() + total_duration_s

                # Expected number of samples ≈ total_duration_s / window_s
                while time.time() < t_end:
                    t_window_start = time.time()
                    active_time = window_s * duty

                    if duty > 0.0:
                        num_ops = max(1, int(active_time / op_time_s))
                        for _ in range(num_ops):
                            _ = a @ b
                        torch.cuda.synchronize(gpu_id)
                    else:
                        num_ops = 0

                    # Sample power once per window
                    P = read_power_W(handle)
                    powers.append(P)

                    elapsed = time.time() - t_window_start
                    sleep_time = max(0.0, window_s - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if len(powers) == 0:
                    print(" no samples, skipping")
                    continue

                avg_P = sum(powers) / len(powers)
                mean = avg_P
                var = sum((p - mean) ** 2 for p in powers) / len(powers)
                std_P = math.sqrt(var)
                delta_idle = avg_P - idle_power_W

                writer.writerow(
                    {
                        "gpu_id": gpu_id,
                        "N": N,
                        "duty": f"{duty:.4f}",
                        "avg_power_W": f"{avg_P:.6f}",
                        "avg_power_minus_idle_W": f"{delta_idle:.6f}",
                        "stddev_power_W": f"{std_P:.6f}",
                        "num_samples": len(powers),
                    }
                )
                f.flush()
                print(
                    f" done: avg={avg_P:.2f} W, Δidle={delta_idle:.2f} W, "
                    f"std={std_P:.2f} W, samples={len(powers)}"
                )

    print(f"\n[DONE] Calibration written to {output_path}")


# ------------------------- CLI -------------------------


def parse_args():
    ap = argparse.ArgumentParser("N×duty GEMM power calibration")
    ap.add_argument("--gpu-id", type=int, default=0, help="GPU index for calibration")
    ap.add_argument(
        "--Ns",
        type=str,
        default="1024,1536,2048,2560,3072,3584,4096",
        help="Comma-separated list of N values (matrix sizes)",
    )
    ap.add_argument(
        "--duties",
        type=str,
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated list of duty values in [0,1]",
    )
    ap.add_argument(
        "--window-s",
        type=float,
        default=0.2,
        help="Window length in seconds for duty cycling",
    )
    ap.add_argument(
        "--total-duration-s",
        type=float,
        default=3.0,
        help="Total duration per (N, duty) point",
    )
    ap.add_argument(
        "--idle-duration-s",
        type=float,
        default=5.0,
        help="Duration to measure idle power before sweep",
    )
    ap.add_argument(
        "--idle-interval-s",
        type=float,
        default=0.2,
        help="Sampling interval for idle power measurement",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="nduty_calibration.csv",
        help="Output CSV path",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available – this script must run on a GPU node."
        )

    nvml_init()
    handle = nvml_get_handle(args.gpu_id)

    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    duties = [float(x) for x in args.duties.split(",") if x.strip()]

    print(f"[INFO] Calibrating GPU {args.gpu_id}")
    print(f"[INFO] Ns = {Ns}")
    print(f"[INFO] duties = {duties}")
    print(
        f"[INFO] window_s={args.window_s}, total_duration_s={args.total_duration_s}, "
        f"idle_duration_s={args.idle_duration_s}"
    )
    print("[INFO] Measuring idle power...")
    idle_power = measure_idle_power(
        handle, duration_s=args.idle_duration_s, interval_s=args.idle_interval_s
    )
    print(f"[INFO] Idle power ≈ {idle_power:.2f} W")

    # Enable TF32 if available for more realistic tensor-core usage on newer GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_nduty_sweep(
        gpu_id=args.gpu_id,
        Ns=Ns,
        duties=duties,
        window_s=args.window_s,
        total_duration_s=args.total_duration_s,
        idle_power_W=idle_power,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
