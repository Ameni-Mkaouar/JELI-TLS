"""
examples/multi_scan_workflow.py
================================
Multi-scan workflow for plots with multiple scanner positions.
This is the correct approach for setups like Lydia's (4 RIEGL scans).

Why process separately?
  Each scan has its own:
  - Distance-decay geometry (ray_length distribution)
  - Solid angle areas per voxel
  - Directional occlusion pattern
  Merging point clouds first would break all three.

Workflow:
  1. Process each scan independently
  2. Merge PAD profiles using weighted averaging
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

from padlad.io       import (load_point_cloud, get_scanner_position,
                               remove_ground, load_stem_mask, save_outputs,
                               merge_profiles)
from padlad.geometry import build_grid
from padlad.pipeline import run_pipeline

# ── Configure your scans here ────────────────────────────────────────────────
SCANS = [
    {"file": "scan1.laz", "scanner": [0.0,   0.0,  1.3]},
    {"file": "scan2.laz", "scanner": [10.0,  0.0,  1.3]},
    {"file": "scan3.laz", "scanner": [10.0, 10.0,  1.3]},
    {"file": "scan4.laz", "scanner": [0.0,  10.0,  1.3]},
]
STEM_FILE    = None          # optional: "stems.txt"
OUTPUT_DIR   = Path("results/multi_scan")
VOXEL_SIZE   = 0.5
BEAM_DIV     = 0.0           # or 7e-5 for Paper [2]

def process_one_scan(scan_info, output_dir, voxel_size, beam_div, stem_file):
    """Process a single scan and return path to its PAD profile CSV."""
    fpath   = scan_info["file"]
    scanner = np.array(scan_info["scanner"])
    stem    = Path(fpath).stem

    print(f"\n{'='*60}")
    print(f"Processing: {fpath}  scanner={scanner}")
    print(f"{'='*60}")

    pts, meta = load_point_cloud(fpath)
    pts, _    = remove_ground(pts, pct=2.0)

    origin, grid_size, _ = build_grid(pts, voxel_size)
    stem_mask = load_stem_mask(stem_file, origin, voxel_size)

    df, best_lidf, best_raw = run_pipeline(
        pts, scanner,
        voxel_d   = voxel_size,
        stem_mask = stem_mask,
        beam_div  = beam_div,
        subsample = 10000,
        verbose   = True,
    )

    scan_out = output_dir / stem
    csv_path, _, _ = save_outputs(df, best_raw, scan_out, stem, beam_div)
    return str(csv_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: process each scan independently
    csv_files = []
    for scan in SCANS:
        csv_path = process_one_scan(
            scan, OUTPUT_DIR, VOXEL_SIZE, BEAM_DIV, STEM_FILE
        )
        csv_files.append(csv_path)

    # Step 2: merge profiles with weighted averaging
    print(f"\n{'='*60}")
    print("Merging profiles …")
    merged_path = str(OUTPUT_DIR / "merged_PAD_profile.csv")
    df_merged = merge_profiles(csv_files, VOXEL_SIZE, merged_path)

    print(f"\n{'─'*50}")
    print(f"  Merged LAI  : {df_merged['LAI_total'].iloc[0]:.3f} m² m⁻²")
    print(f"  Peak PAD    : {df_merged['PAD'].max():.3f} m² m⁻³ "
          f"at z = {df_merged.loc[df_merged['PAD'].idxmax(),'height_m']:.2f} m")
    print(f"  Output      : {merged_path}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
