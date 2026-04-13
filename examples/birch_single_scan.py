"""
examples/birch_single_scan.py
==============================
Reproduces the single-scan analysis from Paper [1] (Mkaouar et al. 2021)
for the Estonian Birch stand.

This example shows the full workflow:
  - Load TLS point cloud
  - Set scanner position (from main MATLAB script: center = [19.93, 50.07, 0.001])
  - Run joint PAD/LIDF inversion
  - Save results
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

from padlad.io       import load_point_cloud, get_scanner_position, remove_ground, save_outputs
from padlad.geometry import build_grid
from padlad.io       import load_stem_mask
from padlad.pipeline import run_pipeline

# ── Configuration (matching MATLAB script) ───────────────────────────────────
INPUT_FILE   = "path/to/tls2.las"      # your .las or .laz file
SCANNER_POS  = [19.93, 50.07, 0.001]  # center = [19.93, 50.07, 0.001] in MATLAB
STEM_FILE    = None                     # or "path/to/birch_stems.txt"
OUTPUT_DIR   = "results/birch_paper1"
VOXEL_SIZE   = 0.5                     # Svox = 0.5 in paper
BEAM_DIV     = 0.0                     # Paper [1]: no beam divergence correction

def main():
    # Load
    pts, meta = load_point_cloud(INPUT_FILE)
    scanner   = get_scanner_position(meta, pts, manual=SCANNER_POS)

    # Ground removal
    pts, _ = remove_ground(pts, pct=2.0)

    # Stem mask (optional)
    origin, grid_size, _ = build_grid(pts, VOXEL_SIZE)
    stem_mask = load_stem_mask(STEM_FILE, origin, VOXEL_SIZE)

    # Run joint inversion
    df, best_lidf, best_raw = run_pipeline(
        pts, scanner,
        voxel_d    = VOXEL_SIZE,
        stem_mask  = stem_mask,
        beam_div   = BEAM_DIV,
        sce_P      = 12,       # P=12 in MATLAB
        sce_n      = 15,       # n=15 in MATLAB
        sce_eps    = 1e-4,     # eps=0.0001 in MATLAB
        subsample  = 10000,
        verbose    = True,
    )

    # Save
    save_outputs(df, best_raw, Path(OUTPUT_DIR), "birch_paper1", BEAM_DIV)

    print(f"\nLAI = {df['LAI_total'].iloc[0]:.3f}")
    print(f"Peak PAD = {df['PAD'].max():.3f} at z = {df.loc[df['PAD'].idxmax(),'height_m']:.2f} m")


if __name__ == "__main__":
    main()
