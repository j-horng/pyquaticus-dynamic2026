#!/usr/bin/env python3
"""
Plot training progress from train.log.

Usage:
  python rl_test/plot_training.py
  python rl_test/plot_training.py --log training/train.log
  python rl_test/plot_training.py --log training/train.log --smooth 5
"""

import argparse
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def parse_log(path):
    """Parse train.log and return lists of (iter, reward, blue_size, red_size)."""
    iters, rewards, blue_sizes, red_sizes = [], [], [], []
    pattern = re.compile(
        r"Iter\s+(\d+):\s+return_mean=([\-\d.]+)"
        r"(?:.*?blue_size=([\d.]+).*?red_size=([\d.]+))?"
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                it = int(m.group(1))
                rew_str = m.group(2)
                try:
                    rew = float(rew_str)
                except ValueError:
                    continue
                iters.append(it)
                rewards.append(rew)
                blue_sizes.append(float(m.group(3)) if m.group(3) else None)
                red_sizes.append(float(m.group(4)) if m.group(4) else None)
    return iters, rewards, blue_sizes, red_sizes


def smooth(values, window):
    """Simple moving average."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from train.log")
    parser.add_argument("--log", default="training/train.log", help="Path to train.log")
    parser.add_argument("--smooth", type=int, default=3, help="Moving-average window for reward (default 3)")
    args = parser.parse_args()

    try:
        iters, rewards, blue_sizes, red_sizes = parse_log(args.log)
    except FileNotFoundError:
        print(f"ERROR: Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    if not iters:
        print("No Iter lines found in log.", file=sys.stderr)
        sys.exit(1)

    has_sizes = any(b is not None for b in blue_sizes)

    # ── Layout ──────────────────────────────────────────────────────────────
    if has_sizes:
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        ax_rew = fig.add_subplot(gs[0, :])   # top row: full width
        ax_blue = fig.add_subplot(gs[1, 0])  # bottom-left
        ax_red = fig.add_subplot(gs[1, 1])   # bottom-right
    else:
        fig, ax_rew = plt.subplots(figsize=(10, 5))

    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    # ── Reward curve ────────────────────────────────────────────────────────
    rewards_np = np.array(rewards, dtype=float)
    smoothed = smooth(rewards_np, args.smooth)

    ax_rew.plot(iters, rewards_np, alpha=0.35, color="#4c8eda", linewidth=1.2, label="Raw return_mean")
    ax_rew.plot(iters, smoothed, color="#1a5fa8", linewidth=2.2, label=f"Smoothed (w={args.smooth})")
    ax_rew.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_rew.set_xlabel("Iteration")
    ax_rew.set_ylabel("Episode Return Mean")
    ax_rew.set_title("Episode Return Mean over Training")
    ax_rew.legend(loc="lower right")
    ax_rew.grid(True, alpha=0.3)

    # ── Team size bar charts ─────────────────────────────────────────────────
    if has_sizes:
        valid_blue = [b for b in blue_sizes if b is not None]
        valid_red = [r for r in red_sizes if r is not None]
        valid_blue_iters = [iters[i] for i, b in enumerate(blue_sizes) if b is not None]
        valid_red_iters = [iters[i] for i, r in enumerate(red_sizes) if r is not None]

        ax_blue.plot(valid_blue_iters, valid_blue, color="#2ca02c", linewidth=2, marker="o", markersize=4)
        ax_blue.set_xlabel("Iteration")
        ax_blue.set_ylabel("Mean Active Agents")
        ax_blue.set_title("Blue Team Size (mean per iter)")
        ax_blue.set_ylim(0, 3.5)
        ax_blue.axhline(2, color="gray", linewidth=0.8, linestyle="--", label="Expected mean (uniform 1-3)")
        ax_blue.legend(fontsize=8)
        ax_blue.grid(True, alpha=0.3)

        ax_red.plot(valid_red_iters, valid_red, color="#d62728", linewidth=2, marker="o", markersize=4)
        ax_red.set_xlabel("Iteration")
        ax_red.set_ylabel("Mean Active Agents")
        ax_red.set_title("Red Team Size (mean per iter)")
        ax_red.set_ylim(0, 3.5)
        ax_red.axhline(2, color="gray", linewidth=0.8, linestyle="--", label="Expected mean (uniform 1-3)")
        ax_red.legend(fontsize=8)
        ax_red.grid(True, alpha=0.3)
    else:
        pass  # No team size data yet — nothing extra to show

    out_path = args.log.replace(".log", "_plot.png").replace("train_plot", "training_plot")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
