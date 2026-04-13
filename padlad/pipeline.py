"""
padlad/pipeline.py
==================
Full PAD/LAD estimation pipeline.
Orchestrates voxelisation, ray traversal, SCE-UA inversion, and output.
"""
import numpy as np
import pandas as pd

from .geometry import build_grid, woo_traverse, voxel_solid_angle_areas
from .sail     import constrain_lidf, NA
from .inversion import compute_pad_voxels, cost_variance, sce_ua


# ─────────────────────────────────────────────────────────────────────────────
#  Voxelisation + ray traversal + transmittance
# ─────────────────────────────────────────────────────────────────────────────

def compute_transmittance(pts: np.ndarray,
                           scanner: np.ndarray,
                           origin: np.ndarray,
                           grid_size: np.ndarray,
                           voxel_d: float,
                           stem_mask: set,
                           subsample: int = 10000,
                           min_beams: int = 5,
                           use_pvlad: bool = True,
                           pe_threshold: float = 1.0) -> tuple:
    """
    Build per-voxel transmittance Tr, path length deltaL,
    zenith angle xth, cos_omg, solid angle area, and
    percentage exploration PE (PVlad enhancement).

    Parameters
    ----------
    use_pvlad    : if True, use range-weighted transmittance (PVlad method)
                   and compute Percentage Exploration for quality masking.
                   If False, use simple hit/pass counts (original Mkaouar method).
    pe_threshold : minimum PE (%) for a voxel to be considered valid.
                   Only used when use_pvlad=True. (default: 1.0%)

    PVlad enhancement (Yin, Cook & Morton 2022):
    --------------------------------------------
    Path Volume per voxel:
        PV_v = sum_k ( path_length_k * w_k )
        where w_k = 1/r_k^2  (range-corrected power weight)

    Percentage Exploration:
        PE_v = PV_v / max(PV) * 100%

    Range-weighted transmittance:
        Tr_w = sum_hit(w) / (sum_hit(w) + sum_pass(w))

    This corrects for range-dependent sampling bias — voxels far from
    the scanner are naturally undersampled, and simple hit/pass counts
    treat all rays equally regardless of their ranging distance.

    Returns
    -------
    centre_vox, Tr, deltaL, xth, cos_omg, angle_area, valid_mask
    """
    shape = tuple(grid_size.tolist())
    mode  = "PVlad (range-weighted)" if use_pvlad else "standard (hit/pass)"
    print(f"\n[2] Voxel grid: {shape}  voxel_d={voxel_d} m  mode={mode}")

    # voxel indices for every point
    idx  = np.floor((pts - origin) / voxel_d).astype(np.int32)
    idx  = np.clip(idx, 0, np.array(shape, dtype=np.int32) - 1)
    vx   = np.unique(idx, axis=0)
    lvx  = len(vx)
    print(f"    Filled voxels: {lvx:,}")

    centre_vox = (vx + 0.5) * voxel_d + origin
    vox2pos    = {tuple(v): i for i, v in enumerate(vx)}

    # ── hit counts ──────────────────────────────────────────────────────────
    hit_count   = np.zeros(lvx, dtype=np.float64)   # unweighted (Mkaouar)
    hit_weight  = np.zeros(lvx, dtype=np.float64)   # range-weighted (PVlad)

    for row in idx:
        key = tuple(row)
        if key in vox2pos:
            vi = vox2pos[key]
            hit_count[vi] += 1.0
            # range weight for this hit
            diff = centre_vox[vi] - scanner
            r2   = max(np.dot(diff, diff), 1e-6)
            hit_weight[vi] += 1.0 / r2

    # ── ray traversal ───────────────────────────────────────────────────────
    n       = len(pts)
    sample  = np.random.choice(n, min(subsample, n), replace=False)
    rays    = pts[sample]
    scale   = n / len(rays)

    pass_count  = np.zeros(lvx, dtype=np.float64)
    pass_weight = np.zeros(lvx, dtype=np.float64)   # PV accumulator
    sum_deltaL  = np.zeros(lvx, dtype=np.float64)
    n_deltaL    = np.zeros(lvx, dtype=np.int32)

    print(f"    Tracing {len(rays):,} rays (scale ×{scale:.1f}) …")

    r_sphere = float(np.max(np.linalg.norm(pts - scanner, axis=1)))

    for pt in rays:
        dn   = pt - scanner
        dist = float(np.linalg.norm(dn))
        if dist < 1e-9:
            continue
        end_pt = scanner + (dn / dist) * r_sphere
        trav, rdists = woo_traverse(scanner, end_pt, origin, voxel_d, shape)

        ray_l = float(np.linalg.norm(end_pt - scanner))
        for k, vox_key in enumerate(trav[:-1]):
            if vox_key not in vox2pos:
                continue
            vi = vox2pos[vox_key]
            pass_count[vi] += scale

            # path length in this voxel
            dl = ray_l * rdists[k] if k == 0 else ray_l * (rdists[k] - rdists[k-1])
            dl = abs(dl)
            sum_deltaL[vi] += dl
            n_deltaL[vi]   += 1

            if use_pvlad:
                # range to voxel centre for weight
                diff = centre_vox[vi] - scanner
                r2   = max(np.dot(diff, diff), 1e-6)
                # PV = path_length * range_weight  (Yin et al. 2022)
                pass_weight[vi] += scale * dl / r2

    mean_deltaL = np.where(n_deltaL > 0, sum_deltaL / n_deltaL, voxel_d)

    # ── transmittance ────────────────────────────────────────────────────────
    if use_pvlad:
        # Range-weighted transmittance (PVlad)
        tot_w = hit_weight + pass_weight
        Tr    = np.where(tot_w > 1e-12,
                         1.0 - hit_weight / tot_w,
                         np.nan)
    else:
        # Original Mkaouar: simple hit/pass ratio
        tot_rays = np.maximum(pass_count + hit_count, 1.0)
        Tr = 1.0 - hit_count / tot_rays

    Tr = np.where(np.isfinite(Tr), np.clip(Tr, 1e-6, 1.0 - 1e-6), np.nan)

    # ── Percentage Exploration (PVlad) ───────────────────────────────────────
    if use_pvlad:
        pv_max = pass_weight.max()
        PE     = np.where(pv_max > 0, pass_weight / pv_max * 100.0, 0.0)
    else:
        PE = np.full(lvx, 100.0)   # all voxels considered fully explored

    # ── validity mask ─────────────────────────────────────────────────────
    invalid = np.zeros(lvx, dtype=bool)
    for sv in stem_mask:
        if sv in vox2pos:
            invalid[vox2pos[sv]] = True

    if use_pvlad:
        invalid |= (PE < pe_threshold)     # PVlad quality filter
    else:
        tot_rays = pass_count + hit_count
        invalid |= (tot_rays < min_beams)  # original beam count filter

    Tr[invalid]          = np.nan
    mean_deltaL[invalid] = np.nan

    # ── per-voxel zenith and cos ─────────────────────────────────────────
    diff    = centre_vox - scanner
    dist_cv = np.linalg.norm(diff, axis=1)
    cos_omg = np.abs(diff[:, 2]) / (dist_cv + 1e-12)
    xth     = np.degrees(np.arccos(np.clip(cos_omg, 0, 1)))

    # ── solid angle areas ─────────────────────────────────────────────────
    print("    Computing solid angle areas …")
    angle_area = voxel_solid_angle_areas(centre_vox, scanner, voxel_d)

    valid = np.isfinite(Tr) & np.isfinite(mean_deltaL)
    if use_pvlad:
        print(f"    Valid voxels: {valid.sum():,}  "
              f"(PE≥{pe_threshold}%: {(PE>=pe_threshold).sum():,} / {lvx:,})")
    else:
        print(f"    Valid voxels: {valid.sum():,}")

    return centre_vox, Tr, mean_deltaL, xth, cos_omg, angle_area, valid


# ─────────────────────────────────────────────────────────────────────────────
#  Zenith binning
# ─────────────────────────────────────────────────────────────────────────────

def make_zenith_bins(xth: np.ndarray,
                     valid: np.ndarray,
                     pas: float = 2.0) -> tuple:
    """
    Group valid voxel indices by zenith angle bin.
    Trims 10° margins at each end — matching MATLAB voxRanges trim.
    Returns (vox_range, med_xth, n_zenith).
    """
    xth_v   = xth[valid]
    xth_min = xth_v.min()
    xth_p   = np.maximum(np.ceil((xth_v - xth_min) / pas).astype(int), 1)
    n_bins  = xth_p.max()

    valid_idx = np.where(valid)[0]
    vox_range = []
    med_xth   = []

    for i in range(1, n_bins + 1):
        local = np.where(xth_p == i)[0]
        if len(local) == 0:
            continue
        vox_range.append(valid_idx[local])
        med_xth.append(xth_min + i * pas - pas / 2.0)

    trim = max(1, int(10.0 / pas))
    if len(vox_range) > 2 * trim:
        vox_range = vox_range[trim:-trim]
        med_xth   = med_xth[trim:-trim]

    return vox_range, np.array(med_xth), len(vox_range)


# ─────────────────────────────────────────────────────────────────────────────
#  Vertical profile assembly
# ─────────────────────────────────────────────────────────────────────────────

def vertical_profile(centre_vox: np.ndarray,
                      Tr: np.ndarray,
                      deltaL: np.ndarray,
                      xth: np.ndarray,
                      cos_omg: np.ndarray,
                      scanner: np.ndarray,
                      best_lidf_raw: np.ndarray,
                      voxel_d: float,
                      beam_div: float,
                      rl: float) -> pd.DataFrame:
    """Compute final PAD per voxel → aggregate to vertical profile."""
    valid  = np.isfinite(Tr) & np.isfinite(deltaL)
    lidf   = constrain_lidf(best_lidf_raw)

    density = compute_pad_voxels(
        Tr[valid], xth[valid], cos_omg[valid],
        deltaL[valid], lidf,
        centre_vox[valid], scanner, beam_div, rl
    )

    z_vals = centre_vox[valid, 2]
    z_min  = np.floor(z_vals.min() / voxel_d) * voxel_d
    z_max  = np.ceil(z_vals.max()  / voxel_d) * voxel_d
    z_bins = np.arange(z_min, z_max + voxel_d, voxel_d)

    rows = []
    for z in z_bins:
        mask = (z_vals >= z - voxel_d/2) & (z_vals < z + voxel_d/2)
        if mask.sum() == 0:
            continue
        finite = density[mask]
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if len(finite) == 0:
            continue
        rows.append({
            "height_m": round(float(z), 3),
            "PAD":      round(float(np.mean(finite)), 5),
            "n_voxels": int(mask.sum())
        })

    df = pd.DataFrame(rows).sort_values("height_m").reset_index(drop=True)
    df["LAI_cumulative"] = (df["PAD"] * voxel_d).cumsum()
    df["LAI_total"]      = df["LAI_cumulative"].max()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(pts: np.ndarray,
                 scanner: np.ndarray,
                 voxel_d: float = 0.5,
                 stem_mask: set = None,
                 subsample: int = 10000,
                 min_beams: int = 5,
                 pas_zenith: float = 2.0,
                 beam_div: float = 0.0,
                 rl: float = 1.0,
                 use_pvlad: bool = True,
                 pe_threshold: float = 1.0,
                 sce_P: int = 12,
                 sce_n: int = 15,
                 sce_eps: float = 1e-4,
                 sce_maxiter: int = 300,
                 seed: int = 42,
                 verbose: bool = True) -> tuple:
    """
    Full PAD/LAD estimation pipeline combining:
      - Mkaouar & Kallel (2021): joint PAD/LIDF inversion via SCE-UA
      - Mkaouar & Kallel (2023): beam divergence correction
      - Yin, Cook & Morton (2022): PVlad path-volume occlusion correction

    Parameters
    ----------
    pts       : (N,3) vegetation point cloud (ground already removed)
    scanner   : (3,) scanner position
    beam_div  : 0 = Paper [1] (no beam div. correction)
                >0 = Paper [2] (with correction, e.g. 7e-5 for birch dataset)
    use_pvlad : True = use PVlad range-weighted transmittance (Yin et al. 2022)
                False = use original simple hit/pass ratio (Mkaouar 2021)
    pe_threshold : minimum Percentage Exploration (%) to consider a voxel valid
                   Only used when use_pvlad=True. Lower = more voxels kept.

    Returns
    -------
    df        : PAD vertical profile DataFrame
    best_lidf : (15,) estimated normalised LIDF
    best_raw  : (15,) raw LIDF for further optimisation
    """
    np.random.seed(seed)
    if stem_mask is None:
        stem_mask = set()

    origin, grid_size, _ = build_grid(pts, voxel_d)

    (centre_vox, Tr, deltaL, xth, cos_omg,
     angle_area, valid) = compute_transmittance(
        pts, scanner, origin, grid_size,
        voxel_d, stem_mask, subsample, min_beams,
        use_pvlad=use_pvlad, pe_threshold=pe_threshold
    )

    vox_range, med_xth, n_zenith = make_zenith_bins(
        xth, valid, pas_zenith)
    if verbose:
        print(f"\n[3] Zenith bins: {n_zenith}  "
              f"({med_xth.min():.1f}°–{med_xth.max():.1f}°)")

    # build local indices for inversion
    vi      = np.where(valid)[0]
    pos_map = {g: l for l, g in enumerate(vi)}
    vr_local = [np.array([pos_map[g] for g in grp if g in pos_map])
                for grp in vox_range]
    vr_local = [g for g in vr_local if len(g) > 0]

    Tr_v   = Tr[vi];        xth_v  = xth[vi]
    dL_v   = deltaL[vi];    cos_v  = cos_omg[vi]
    cv_v   = centre_vox[vi]

    def _cost(raw):
        return cost_variance(raw, Tr_v, xth_v, cos_v, dL_v,
                             vr_local, cv_v, scanner, beam_div, rl)

    best_raw, best_cost = sce_ua(
        _cost, n_var=NA, P=sce_P, n=sce_n,
        eps=sce_eps, max_iter=sce_maxiter,
        seed=seed, verbose=verbose
    )

    best_lidf = constrain_lidf(best_raw)
    if verbose:
        from .sail import LITAB
        print(f"\n[5] Estimated LIDF:")
        for ang, val in zip(LITAB, best_lidf):
            bar = '█' * int(val * 60)
            print(f"    {ang:5.1f}°  {val:.4f}  {bar}")

    df = vertical_profile(
        centre_vox, Tr, deltaL, xth, cos_omg, scanner,
        best_raw, voxel_d, beam_div, rl
    )

    if verbose:
        print(f"\n{'─'*52}")
        print(f"  LAI total  : {df['LAI_total'].iloc[0]:.3f} m² m⁻²")
        print(f"  Peak PAD   : {df['PAD'].max():.3f} m² m⁻³ "
              f"at z = {df.loc[df['PAD'].idxmax(),'height_m']:.2f} m")
        print(f"  Beam div.  : "
              f"{'enabled ε='+str(beam_div) if beam_div>0 else 'disabled'}")
        print(f"{'─'*52}")

    return df, best_lidf, best_raw
