#!/usr/bin/env python3
"""
calibrate_gpu_burn.py – GEMM-based power calibrator for a single GPU
====================================================================

This script lets you:
  * Run a smooth ramp up/down in power using GEMM-based synthetic load
  * Sweep duty cycles (0.0–1.0) and record average power for each duty

The goal is to learn a mapping duty -> power for your GPU so you can
design startup/shutdown ramps and checkpoint-fill burns for training.

Requires:
  pip install pynvml

Example usages
--------------

# 1) Simple ramp up over 30s, then ramp down over 30s (for eyeballing)
python calibrate_gpu_burn.py --gpu 0 --mode ramp --ramp_seconds 30

# 2) Duty sweep from 0.1 to 1.0 in 0.1 steps, 5s per point
python calibrate_gpu_burn.py --gpu 0 --mode sweep \
    --duty_min 0.1 --duty_max 1.0 --duty_step 0.1 \
    --dwell_seconds 5.0 \
    --csv duty_sweep_gpu0.csv
"""

import argparse
import csv
import math
import time

import torch

try:
    import pynvml
except ImportError:
    pynvml = None
    raise SystemExit("pynvml not installed. Run: pip install pynvml")


# ---------------------------------------------------------------------------
# Burner: GEMM-based synthetic load with calibrated op time + duty-cycle
# ---------------------------------------------------------------------------


class GPUPowerBurner:
    """
    GEMM-based synthetic load generator with duty-cycle control.

    - Allocates a square working set A, B on the target GPU.
    - Calibrates average time per matmul.
    - Exposes:
        * run_window(duty, window_s)
        * ramp(duration_s, direction)
    """

    def __init__(
        self,
        gpu_id: int,
        burn_mem_fraction: float = 0.1,
        max_matrix_dim: int = 4096,
        dtype: torch.dtype = torch.float32,
    ):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.dtype = dtype

        total_mem_bytes = torch.cuda.get_device_properties(gpu_id).total_memory
        burn_bytes = int(total_mem_bytes * burn_mem_fraction)

        # Approximate: 2 * N^2 * sizeof(elem) live during matmul
        elem_bytes = torch.finfo(self.dtype).bits // 8
        N = int(math.sqrt(burn_bytes / (2.0 * elem_bytes)))
        N = max(128, min(max_matrix_dim, (N // 64) * 64))  # nice multiple

        print(
            f"[BURN] GPU {gpu_id}: total_mem={total_mem_bytes / 1024**3:.2f} GB, "
            f"burn_mem_fraction={burn_mem_fraction}, N={N}"
        )

        self.N = N
        self._alloc_buffers()
        self._calibrate_op_time()

    def _alloc_buffers(self):
        torch.cuda.set_device(self.gpu_id)
        self.A = torch.randn(self.N, self.N, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.N, self.N, device=self.device, dtype=self.dtype)

    def _calibrate_op_time(self, num_ops: int = 10):
        # Warm-up
        for _ in range(3):
            _ = self.A @ self.B
        torch.cuda.synchronize(self.gpu_id)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_ops):
            _ = self.A @ self.B
        end.record()
        torch.cuda.synchronize(self.gpu_id)

        ms = start.elapsed_time(end)
        self.op_time_s = (ms / 1000.0) / num_ops
        print(f"[CAL] GPU {self.gpu_id}: avg matmul time = {self.op_time_s:.6f} s")

    def run_window(self, duty_cycle: float, window_s: float):
        """
        Run synthetic load for one time-window.

        duty_cycle in [0, 1]:
          0.0 -> no GEMMs (just sleep)
          1.0 -> GEMMs for entire window (best effort, may slightly spill)

        window_s: total duration of this "control interval".
        """
        duty_cycle = max(0.0, min(1.0, duty_cycle))
        if window_s <= 0.0:
            return

        active_time = window_s * duty_cycle
        if active_time <= 0.0:
            time.sleep(window_s)
            return

        num_ops = max(1, int(active_time / self.op_time_s))

        # Active region: do num_ops GEMMs
        for _ in range(num_ops):
            _ = self.A @ self.B
        torch.cuda.synchronize(self.gpu_id)

        elapsed = num_ops * self.op_time_s
        sleep_time = max(0.0, window_s - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def ramp(self, duration_s: float, direction: str = "up", window_s: float = 0.1):
        """
        Cosine-shaped ramp up/down in duty from 0 -> max (up) or max -> 0 (down).

        direction: "up" or "down"
        duration_s: total duration
        window_s: per-step window length
        """
        if duration_s <= 0:
            return

        num_windows = max(1, int(duration_s / window_s))
        print(
            f"[RAMP] GPU {self.gpu_id}: direction={direction}, "
            f"duration={duration_s}s, windows={num_windows}, window_s={window_s}"
        )

        for i in range(num_windows):
            x = i / max(1, num_windows - 1)  # 0..1
            if direction == "up":
                # smooth 0 -> 1
                frac = 0.5 * (1.0 - math.cos(math.pi * x))
            else:
                # smooth 1 -> 0
                frac = 0.5 * (1.0 + math.cos(math.pi * x))

            self.run_window(frac, window_s)


# ---------------------------------------------------------------------------
# NVML helpers
# ---------------------------------------------------------------------------


def init_nvml(gpu_id: int):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    name = pynvml.nvmlDeviceGetName(handle)
    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    # Handle both bytes and str return types from pynvml
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    print(f"[NVML] GPU {gpu_id}: {name} (power limit ~{power_limit:.1f} W)")
    return handle


def read_power_w(handle) -> float:
    """Return instantaneous power in watts."""
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_ramp_mode(args):
    handle = init_nvml(args.gpu)
    burner = GPUPowerBurner(
        gpu_id=args.gpu,
        burn_mem_fraction=args.burn_mem_fraction,
        max_matrix_dim=args.max_matrix_dim,
    )

    # Idle baseline
    print("[RAMP] Measuring idle power for 5 seconds...")
    idle_samples = []
    t0 = time.time()
    while time.time() - t0 < 5.0:
        idle_samples.append(read_power_w(handle))
        time.sleep(0.2)
    idle_avg = sum(idle_samples) / len(idle_samples)
    print(f"[RAMP] Idle average power: {idle_avg:.2f} W")

    # Ramp up
    print("[RAMP] Starting ramp UP...")
    burner.ramp(duration_s=args.ramp_seconds, direction="up", window_s=args.window_s)

    # Steady at top for a bit
    print("[RAMP] Holding steady at full duty for 5 seconds...")
    t1 = time.time()
    while time.time() - t1 < 5.0:
        burner.run_window(1.0, args.window_s)

    # Ramp down
    print("[RAMP] Starting ramp DOWN...")
    burner.ramp(duration_s=args.ramp_seconds, direction="down", window_s=args.window_s)

    print("[RAMP] Done.")


def run_sweep_mode(args):
    handle = init_nvml(args.gpu)
    burner = GPUPowerBurner(
        gpu_id=args.gpu,
        burn_mem_fraction=args.burn_mem_fraction,
        max_matrix_dim=args.max_matrix_dim,
    )

    # Baseline idle
    print("[SWEEP] Measuring idle baseline...")
    idle_samples = []
    t0 = time.time()
    while time.time() - t0 < args.dwell_seconds:
        idle_samples.append(read_power_w(handle))
        time.sleep(args.sample_period)
    idle_avg = sum(idle_samples) / len(idle_samples)
    print(f"[SWEEP] Idle average over {args.dwell_seconds}s: {idle_avg:.2f} W")

    # Prepare CSV
    out_path = args.csv or f"duty_sweep_gpu{args.gpu}.csv"
    print(f"[SWEEP] Writing results to: {out_path}")
    f = open(out_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "gpu_id",
            "duty",
            "avg_power_W",
            "avg_power_minus_idle_W",
            "stddev_power_W",
            "num_samples",
        ]
    )

    duty = args.duty_min
    while duty <= args.duty_max + 1e-8:
        print(f"[SWEEP] duty={duty:.3f}, dwell={args.dwell_seconds}s")
        samples = []
        t_start = time.time()
        while time.time() - t_start < args.dwell_seconds:
            burner.run_window(duty, args.window_s)
            p = read_power_w(handle)
            samples.append(p)
            time.sleep(args.sample_period)

        if samples:
            avg_p = sum(samples) / len(samples)
            var = sum((x - avg_p) ** 2 for x in samples) / len(samples)
            std_p = math.sqrt(var)
        else:
            avg_p, std_p = float("nan"), float("nan")

        delta = avg_p - idle_avg
        print(
            f"    avg={avg_p:.2f} W, delta={delta:.2f} W over idle, "
            f"samples={len(samples)}, std={std_p:.2f} W"
        )
        writer.writerow(
            [
                args.gpu,
                f"{duty:.4f}",
                f"{avg_p:.6f}",
                f"{delta:.6f}",
                f"{std_p:.6f}",
                len(samples),
            ]
        )
        f.flush()

        duty += args.duty_step

    f.close()
    print("[SWEEP] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("GPU GEMM-based power calibration tool")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (as in nvidia-smi)")
    p.add_argument(
        "--mode",
        choices=["ramp", "sweep"],
        default="sweep",
        help="ramp: smooth 0->1->0; sweep: duty sweep with CSV output",
    )

    # Shared burner params
    p.add_argument(
        "--burn_mem_fraction",
        type=float,
        default=0.1,
        help="Fraction of GPU memory used for GEMM buffers",
    )
    p.add_argument(
        "--max_matrix_dim",
        type=int,
        default=4096,
        help="Cap on GEMM dimension N to avoid absurd allocations",
    )
    p.add_argument(
        "--window_s",
        type=float,
        default=0.1,
        help="Control window length for duty cycle (seconds)",
    )

    # Ramp mode params
    p.add_argument(
        "--ramp_seconds",
        type=float,
        default=30.0,
        help="Duration of ramp up/down in seconds (for ramp mode)",
    )

    # Sweep mode params
    p.add_argument("--duty_min", type=float, default=0.1)
    p.add_argument("--duty_max", type=float, default=1.0)
    p.add_argument("--duty_step", type=float, default=0.1)
    p.add_argument(
        "--dwell_seconds",
        type=float,
        default=5.0,
        help="How long to dwell at each duty level",
    )
    p.add_argument(
        "--sample_period",
        type=float,
        default=0.1,
        help="How often to sample NVML power (seconds)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV path (default: duty_sweep_gpu<id>.csv)",
    )

    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    if args.mode == "ramp":
        run_ramp_mode(args)
    else:
        run_sweep_mode(args)

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
