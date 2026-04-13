"""
padlad/io.py
============
Point cloud loading, result saving, and multi-scan profile averaging.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import laspy
    _LASPY = True
except ImportError:
    _LASPY = False

from .sail import LITAB, constrain_lidf


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_txt_robust(filepath: str) -> np.ndarray:
    """
    Robustly load any common TLS text export format:
      - Space, tab, or comma delimited
      - Comment lines starting with # skipped
      - Extra columns beyond XYZ ignored (e.g. Intensity, R, G, B, ...)
      - Empty lines skipped

    Supports: .txt .csv .pts .xyz .asc (all treated identically)
    """
    rows = []
    delimiter = None

    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # auto-detect delimiter from first data line
            if delimiter is None:
                if ',' in line:
                    delimiter = ','
                elif '\t' in line:
                    delimiter = '\t'
                else:
                    delimiter = None     # whitespace split
            try:
                parts = line.split(delimiter) if delimiter else line.split()
                if len(parts) < 3:
                    continue
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue   # skip header/non-numeric lines silently

    if not rows:
        raise ValueError(f"No numeric XYZ data found in {filepath}")
    return np.array(rows, dtype=np.float64)


def load_point_cloud(filepath: str) -> tuple:
    """
    Load a TLS point cloud from any common format.

    Supported formats
    -----------------
    .las / .laz  : LAS/LAZ binary (requires laspy[lazrs])
    .txt         : space, tab, or comma delimited XYZ (extra columns ignored)
    .csv         : comma delimited XYZ
    .pts         : FARO .pts format (XYZ + optional columns)
    .xyz         : space-delimited XYZ
    .asc         : space-delimited XYZ
    .npy         : NumPy binary array, shape (N,3) or (N,>3)

    Comment lines starting with # or // are skipped automatically.
    Returns (N,3) XYZ array and metadata (laspy object or None).
    """
    p   = Path(filepath)
    ext = p.suffix.lower()
    print(f"\n[1] Loading: {p.name}")

    if ext in (".las", ".laz"):
        if not _LASPY:
            raise ImportError("pip install laspy[lazrs]")
        las  = laspy.read(str(p))
        pts  = np.stack([np.array(las.x),
                         np.array(las.y),
                         np.array(las.z)], axis=1)
        meta = las
    elif ext in (".txt", ".csv", ".pts", ".xyz", ".asc"):
        pts  = _load_txt_robust(str(p))
        meta = None
    elif ext == ".npy":
        raw  = np.load(str(p))
        pts  = raw[:, :3] if raw.ndim == 2 else raw
        meta = None
    else:
        raise ValueError(
            f"Unsupported format: {ext}\n"
            f"Supported: .las, .laz, .txt, .csv, .pts, .xyz, .asc, .npy"
        )

    print(f"    {len(pts):,} points | "
          f"X∈[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"Z∈[{pts[:,2].min():.2f},{pts[:,2].max():.2f}] m")
    return pts, meta


def get_scanner_position(meta, pts: np.ndarray,
                          manual=None) -> np.ndarray:
    """
    Scanner position priority:
      1. User-supplied manual=[x,y,z]
      2. sensor_origin from LAS header
      3. Centroid of lowest 1% of points + 1.3 m (fallback)
    """
    if manual is not None:
        pos = np.array(manual, dtype=float)
        print(f"    Scanner (user): {pos}")
        return pos

    if meta is not None and _LASPY:
        try:
            o = meta.header.sensor_origin
            if o is not None and not all(v == 0 for v in o[:3]):
                pos = np.array([float(o[0]), float(o[1]), float(o[2])])
                print(f"    Scanner (header): {pos}")
                return pos
        except Exception:
            pass

    low = pts[:, 2] < np.percentile(pts[:, 2], 1)
    pos = np.array([np.median(pts[low, 0]),
                    np.median(pts[low, 1]),
                    pts[:, 2].min() + 1.3])
    warnings.warn("Scanner position estimated from data. "
                  "Use --scanner_x/y/z for accuracy.", UserWarning)
    print(f"    Scanner (estimated): {pos.round(3)}")
    return pos


def remove_ground(pts: np.ndarray, pct: float = 2.0) -> tuple:
    """Height-threshold ground removal. Returns (vegetation_pts, ground_z)."""
    gz  = np.percentile(pts[:, 2], pct)
    veg = pts[pts[:, 2] > gz + 0.05]
    print(f"    Ground ≈ {gz:.2f} m | veg. points: {len(veg):,}")
    return veg, gz


def load_stem_mask(stem_file: str, origin: np.ndarray,
                   voxel_d: float) -> set:
    """Load stem point cloud and return set of voxel index tuples."""
    if stem_file is None:
        return set()
    sp  = np.loadtxt(stem_file)[:, :3]
    idx = np.floor((sp - origin) / voxel_d).astype(np.int32)
    return {tuple(r) for r in idx}


# ─────────────────────────────────────────────────────────────────────────────
#  Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame,
                 best_lidf_raw: np.ndarray,
                 out_dir: Path,
                 stem: str,
                 beam_div: float) -> tuple:
    """
    Save PAD profile CSV, LIDF CSV, and three-panel figure.
    Returns (csv_path, lidf_path, fig_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_dir / f"{stem}_PAD_profile.csv"
    lidf_path = out_dir / f"{stem}_LIDF.csv"
    fig_path  = out_dir / f"{stem}_PAD_profile.png"

    df.to_csv(csv_path, index=False)

    lidf_out = constrain_lidf(best_lidf_raw)
    pd.DataFrame({"angle_deg": LITAB,
                  "LIDF": lidf_out}).to_csv(lidf_path, index=False)

    _make_figure(df, lidf_out, fig_path, stem, beam_div)

    print(f"\n    PAD profile → {csv_path}")
    print(f"    LIDF        → {lidf_path}")
    print(f"    Figure      → {fig_path}")
    return csv_path, lidf_path, fig_path


def _make_figure(df, lidf_out, fig_path, stem, beam_div):
    tag  = " (beam div. corrected)" if beam_div > 0 else ""
    year = "2023" if beam_div > 0 else "2021"

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    fig.suptitle(
        f"PAD/LAD — {stem}{tag}\n"
        f"Mkaouar & Kallel {year} | LAI = {df['LAI_total'].iloc[0]:.3f}",
        fontsize=11, fontweight="bold"
    )

    dz = df["height_m"].diff().median()
    axes[0].barh(df["height_m"], df["PAD"], height=dz*0.85,
                 color="#1565C0", alpha=0.75, edgecolor="none")
    axes[0].set_xlabel("PAD (m² m⁻³)")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("Plant Area Density")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1].barh(LITAB, lidf_out, height=4,
                 color="#2E7D32", alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("Frequency")
    axes[1].set_title("Estimated LIDF (15-bin)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    axes[2].plot(df["LAI_cumulative"], df["height_m"],
                 color="#1B5E20", linewidth=2)
    axes[2].fill_betweenx(df["height_m"], 0, df["LAI_cumulative"],
                           alpha=0.15, color="#388E3C")
    axes[2].axvline(df["LAI_total"].iloc[0], color="gray",
                    linestyle="--", linewidth=1,
                    label=f"LAI={df['LAI_total'].iloc[0]:.2f}")
    axes[2].set_xlabel("Cumulative LAI (m² m⁻²)")
    axes[2].set_title("Cumulative LAI")
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-scan profile averaging
# ─────────────────────────────────────────────────────────────────────────────

def merge_profiles(csv_files: list,
                   voxel_d: float,
                   out_path: str = None) -> pd.DataFrame:
    """
    Weighted average of PAD profiles from multiple single-scan runs.
    Weight = number of valid voxels per height layer.

    This is the correct workflow for Lydia's multi-position setup:
      - Process each scan independently with its own scanner position
      - Average the resulting profiles here

    Parameters
    ----------
    csv_files : list of PAD profile CSV paths
    voxel_d   : voxel size (m) — used for height bin alignment
    out_path  : optional output CSV path

    Returns
    -------
    merged DataFrame with weighted-mean PAD per height layer
    """
    profiles = [pd.read_csv(f) for f in csv_files]

    # collect all heights
    all_heights = sorted(set(
        round(h, 3)
        for df in profiles
        for h in df["height_m"]
    ))

    rows = []
    for z in all_heights:
        pads = []; weights = []
        for df in profiles:
            row = df[abs(df["height_m"] - z) < voxel_d/2]
            if len(row) > 0 and not np.isnan(row["PAD"].values[0]):
                pads.append(float(row["PAD"].values[0]))
                weights.append(float(row["n_voxels"].values[0]))

        if len(pads) == 0:
            continue
        w = np.array(weights)
        pad_avg = float(np.average(pads, weights=w))
        rows.append({
            "height_m":   z,
            "PAD":        round(pad_avg, 5),
            "n_scans":    len(pads),
            "n_voxels":   int(np.sum(w))
        })

    df_merged = pd.DataFrame(rows).sort_values("height_m").reset_index(drop=True)
    df_merged["LAI_cumulative"] = (df_merged["PAD"] * voxel_d).cumsum()
    df_merged["LAI_total"]      = df_merged["LAI_cumulative"].max()

    if out_path:
        df_merged.to_csv(out_path, index=False)
        print(f"Merged profile → {out_path}")

    print(f"Merged {len(csv_files)} scans | "
          f"LAI = {df_merged['LAI_total'].iloc[0]:.3f}")
    return df_merged
