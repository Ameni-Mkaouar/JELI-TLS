"""
examples/beam_divergence_correction.py
=======================================
Demonstrates the beam divergence correction from Paper [2].

The correction accounts for the fact that the laser footprint grows
with distance: a voxel at 20m is sampled by a much wider beam than
one at 2m, biasing the transmittance estimate.

Formula (VarLAI_corr2.m):
    lidar_tab = tan(ε/2) * ||voxel_centre - scanner||
    cor = (rl² * G) / ((rl + lidar_tab)(rl * G + lidar_tab))
    PAD_corrected = PAD * cor

ε = beam half-divergence angle (instrument-specific, see table below)
rl = range parameter (default 1.0)

Beam half-divergence by scanner:
    RIEGL VZ-400 / VZ-400i : 3.5e-4 rad
    Leica ScanStation P40   : 2.3e-4 rad
    Faro Focus 3D           : 1.9e-4 rad
    Paper [2] birch dataset : 7e-5 rad
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from padlad.io       import load_point_cloud, get_scanner_position, remove_ground, save_outputs
from padlad.geometry import build_grid
from padlad.io       import load_stem_mask
from padlad.pipeline import run_pipeline

INPUT_FILE  = "path/to/plot.laz"
SCANNER_POS = [19.93, 50.07, 0.001]
OUTPUT_DIR  = Path("results/beam_div_comparison")
VOXEL_SIZE  = 0.5
BEAM_DIVS   = {
    "No correction (Paper [1])": 0.0,
    "Birch dataset (7e-5)":      7e-5,
    "RIEGL VZ-400 (3.5e-4)":    3.5e-4,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pts, meta = load_point_cloud(INPUT_FILE)
    scanner   = get_scanner_position(meta, pts, manual=SCANNER_POS)
    pts, _    = remove_ground(pts)
    origin, grid_size, _ = build_grid(pts, VOXEL_SIZE)

    results = {}
    for label, beam_div in BEAM_DIVS.items():
        print(f"\n{'='*55}")
        print(f"  {label}  (ε={beam_div})")
        print(f"{'='*55}")
        df, _, best_raw = run_pipeline(
            pts, scanner,
            voxel_d  = VOXEL_SIZE,
            beam_div = beam_div,
            subsample = 10000,
            verbose   = True,
        )
        results[label] = df
        stem = f"beam_div_{str(beam_div).replace('.','p').replace('-','m')}"
        save_outputs(df, best_raw, OUTPUT_DIR, stem, beam_div)

    # Comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    colors = ["#1565C0", "#2E7D32", "#C62828"]

    for (label, df), color in zip(results.items(), colors):
        dz = df["height_m"].diff().median()
        axes[0].barh(df["height_m"], df["PAD"],
                     height=dz*0.7, alpha=0.6,
                     label=label, color=color, edgecolor="none")
        axes[1].plot(df["LAI_cumulative"], df["height_m"],
                     color=color, linewidth=2, label=label)

    axes[0].set_xlabel("PAD (m² m⁻³)")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("PAD profiles")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1].set_xlabel("Cumulative LAI (m² m⁻²)")
    axes[1].set_title("Cumulative LAI")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Effect of beam divergence correction", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "beam_div_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison figure → {fig_path}")


if __name__ == "__main__":
    main()
