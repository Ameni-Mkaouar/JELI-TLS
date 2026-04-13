#!/usr/bin/env python3
"""
pad_lad  — command-line tool for PAD/LAD estimation from TLS point clouds
Mkaouar & Kallel (2021, 2023)

Usage
-----
# Paper [1] — no beam divergence correction:
python cli.py --input plot.laz --voxel_size 0.5

# Paper [2] — with beam divergence correction:
python cli.py --input plot.laz --voxel_size 0.5 --beam_divergence 7e-5

# Multi-scan: process each scan, then merge profiles:
python cli.py --input scan1.laz --scanner_x 0 --scanner_y 0 --scanner_z 1.3 --output results/scan1/
python cli.py --input scan2.laz --scanner_x 5 --scanner_y 0 --scanner_z 1.3 --output results/scan2/
python cli.py --merge results/scan1/scan1_PAD_profile.csv results/scan2/scan2_PAD_profile.csv --output results/merged.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="PAD/LAD joint estimator — Mkaouar & Kallel 2021/2023",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output
    parser.add_argument("--input",    default=None,
                        help=".las, .laz, .txt, .csv, .pts, .xyz, .asc, or .npy point cloud")
    parser.add_argument("--output",   default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--merge",    nargs="+", default=None,
                        help="Merge multiple PAD profile CSVs into one")

    # Scanner
    parser.add_argument("--scanner_x", type=float, default=None)
    parser.add_argument("--scanner_y", type=float, default=None)
    parser.add_argument("--scanner_z", type=float, default=None)

    # Voxel grid
    parser.add_argument("--voxel_size", type=float, default=0.5,
                        help="Voxel edge (m). Paper used 0.5 m. (default: 0.5)")

    # Physics
    parser.add_argument("--beam_divergence", type=float, default=0.0,
                        help="Beam half-angle (rad). 0=Paper[1], 7e-5=Paper[2] birch. "
                             "See README for scanner-specific values.")
    parser.add_argument("--rl", type=float, default=1.0,
                        help="Range parameter for beam div. correction (default: 1.0)")

    # Preprocessing
    parser.add_argument("--stem_cloud",  default=None,
                        help=".txt stem point cloud for trunk masking")
    parser.add_argument("--ground_pct",  type=float, default=2.0,
                        help="Percentile for ground removal (default: 2.0)")

    # Performance
    parser.add_argument("--subsample",   type=int, default=10000,
                        help="Rays to trace. More=accurate/slower. (default: 10000)")
    parser.add_argument("--min_beams",   type=int, default=5,
                        help="Min rays per voxel for valid PAD — used when "
                             "--no_pvlad is set (default: 5)")
    parser.add_argument("--no_pvlad",    action="store_true",
                        help="Disable PVlad range-weighted transmittance. "
                             "Uses original Mkaouar 2021 simple hit/pass counts.")
    parser.add_argument("--pe_threshold", type=float, default=1.0,
                        help="Min Percentage Exploration %% for valid voxel "
                             "(PVlad mode only, default: 1.0)")
    parser.add_argument("--pas_zenith",  type=float, default=2.0,
                        help="Zenith bin width degrees (default: 2.0)")

    # SCE-UA
    parser.add_argument("--sce_P",       type=int,   default=12,
                        help="SCE-UA complexes (default: 12)")
    parser.add_argument("--sce_n",       type=int,   default=15,
                        help="SCE-UA simplex parameter (default: 15)")
    parser.add_argument("--sce_eps",     type=float, default=1e-4,
                        help="SCE-UA convergence threshold (default: 1e-4)")
    parser.add_argument("--sce_maxiter", type=int,   default=300)
    parser.add_argument("--seed",        type=int,   default=42)

    args = parser.parse_args()

    # ── Merge mode ───────────────────────────────────────────────────────────
    if args.merge:
        from padlad.io import merge_profiles
        df = merge_profiles(args.merge, args.voxel_size, args.output)
        print(f"\nMerged {len(args.merge)} profiles → LAI = {df['LAI_total'].iloc[0]:.3f}")
        return

    # ── Single scan mode ─────────────────────────────────────────────────────
    if args.input is None:
        parser.error("--input is required (or use --merge)")

    from padlad.io import (load_point_cloud, get_scanner_position,
                            remove_ground, load_stem_mask, save_outputs)
    from padlad.geometry import build_grid
    from padlad.pipeline import run_pipeline

    np.random.seed(args.seed)

    # Load
    pts, meta = load_point_cloud(args.input)
    manual    = ([args.scanner_x, args.scanner_y, args.scanner_z]
                 if args.scanner_x is not None else None)
    scanner   = get_scanner_position(meta, pts, manual)

    # Preprocess
    pts, _ = remove_ground(pts, args.ground_pct)
    origin, grid_size, _ = build_grid(pts, args.voxel_size)
    stem_mask = load_stem_mask(args.stem_cloud, origin, args.voxel_size)
    if stem_mask:
        print(f"    Stem mask: {len(stem_mask)} stem voxels excluded")

    # Check single-scan constraint
    if args.scanner_x is None:
        print("\n  ⚠  Scanner position not supplied — estimated from data.")
        print("     For multi-scan plots, process each scan separately and")
        print("     use --merge to combine profiles. See README for details.\n")

    # Run
    df, best_lidf, best_raw = run_pipeline(
        pts, scanner,
        voxel_d     = args.voxel_size,
        stem_mask   = stem_mask,
        subsample   = args.subsample,
        min_beams   = args.min_beams,
        pas_zenith  = args.pas_zenith,
        beam_div    = args.beam_divergence,
        rl          = args.rl,
        use_pvlad   = not args.no_pvlad,
        pe_threshold = args.pe_threshold,
        sce_P       = args.sce_P,
        sce_n       = args.sce_n,
        sce_eps     = args.sce_eps,
        sce_maxiter = args.sce_maxiter,
        seed        = args.seed,
        verbose     = True
    )

    # Save
    stem_name = Path(args.input).stem
    save_outputs(df, best_raw, Path(args.output), stem_name, args.beam_divergence)


if __name__ == "__main__":
    main()
