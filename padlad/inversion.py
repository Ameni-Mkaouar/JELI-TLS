"""
padlad/inversion.py
===================
SCE-UA optimiser, cost function, and PAD per voxel.
Faithful translation of simplex_tst.m, VarLAI_corr2.m, main MATLAB script.
"""
import numpy as np
from .sail import new_ks_wo, constrain_lidf, constrain_raw, NA


# ─────────────────────────────────────────────────────────────────────────────
#  PAD per voxel  (VarLAI_corr2.m core)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pad_voxels(Tr: np.ndarray,
                        xth: np.ndarray,
                        cos_omg: np.ndarray,
                        deltaL: np.ndarray,
                        lidf: np.ndarray,
                        centre_vox: np.ndarray,
                        scanner: np.ndarray,
                        beam_div: float = 0.0,
                        rl: float = 1.0) -> np.ndarray:
    """
    PAD = -log(Tr) / (G * deltaL)   with G = ks * cos(theta)

    Optional beam divergence correction (Paper [2] / VarLAI_corr2.m):
        lidar_tab = tan(epsilon/2) * ||vox_centre - scanner||
        cor = (rl² * G) / ((rl + lidar_tab)(rl*G + lidar_tab))
        PAD_cor = PAD * cor

    Parameters
    ----------
    lidf     : normalised 15-bin LIDF (output of constrain_lidf)
    beam_div : beam half-divergence angle in radians (0 = no correction)
    rl       : range parameter (rl in VarLAI_corr2.m)
    """
    ldx = len(Tr)
    k1  = np.zeros(ldx)
    for i in range(ldx):
        ks, _ = new_ks_wo(float(xth[i]), lidf)
        k1[i] = ks

    gs      = k1 * cos_omg
    denom   = gs * deltaL
    denom   = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    density = -np.log(np.clip(Tr, 1e-9, 1.0)) / denom

    if beam_div > 0.0:
        p_cv      = np.linalg.norm(centre_vox - scanner, axis=1)
        lidar_tab = np.tan(beam_div / 2.0) * p_cv
        cor = (rl**2 * gs) / (
            (rl + lidar_tab) * (rl * gs + lidar_tab) + 1e-12
        )
        density = density * cor

    return density


# ─────────────────────────────────────────────────────────────────────────────
#  Cost function  (VarLAI_test2 / VarLAI_corr2 — variance of density_moy)
# ─────────────────────────────────────────────────────────────────────────────

def cost_variance(lidf_raw: np.ndarray,
                  Tr: np.ndarray,
                  xth: np.ndarray,
                  cos_omg: np.ndarray,
                  deltaL: np.ndarray,
                  vox_range: list,
                  centre_vox: np.ndarray,
                  scanner: np.ndarray,
                  beam_div: float,
                  rl: float) -> float:
    """
    Cost = var(mean_PAD per zenith bin).
    Faithful translation of VarLAI_corr2.m / VarLAI_test.m.
    Minimised by SCE-UA to find the LIDF that makes PAD consistent
    across all viewing directions.
    """
    lidf    = constrain_lidf(lidf_raw)
    density = compute_pad_voxels(Tr, xth, cos_omg, deltaL, lidf,
                                  centre_vox, scanner, beam_div, rl)
    n_bins  = len(vox_range)
    density_moy = np.zeros(n_bins)
    for i, idx_list in enumerate(vox_range):
        if len(idx_list) > 0:
            vals = density[idx_list]
            finite = vals[np.isfinite(vals) & (vals > 0)]
            if len(finite) > 0:
                density_moy[i] = float(np.mean(finite))
    finite_moy = density_moy[np.isfinite(density_moy) & (density_moy > 0)]
    if len(finite_moy) < 2:
        return 1e9
    return float(np.var(finite_moy))


# ─────────────────────────────────────────────────────────────────────────────
#  Simplex step  (simplex_tst.m — faithful translation)
# ─────────────────────────────────────────────────────────────────────────────

def simplex_step(vertices: np.ndarray,
                 f: np.ndarray,
                 cost_fn) -> tuple:
    """
    One Nelder-Mead simplex evolution step.
    Faithful translation of simplex_tst.m (Ameni Mkaouar).

    Operations in order: sort → reflect → extend or contract → shrink.
    LIDF constraint applied after each candidate generation.
    """
    order    = np.argsort(f)
    vertices = vertices[order].copy()
    f        = f[order].copy()
    n_v      = len(vertices)
    if n_v < 2:
        return vertices, f

    g = np.mean(vertices[:-1], axis=0)   # centroid of n-1 best

    # reflection: r = 2g - worst
    r    = constrain_raw(2.0*g - vertices[-1])
    ct_r = cost_fn(r)

    if ct_r <= f[-1]:
        # extension: e = 3g - 2*worst
        e    = constrain_raw(3.0*g - 2.0*vertices[-1])
        ct_e = cost_fn(e)
        if ct_e <= ct_r:
            vertices[-1] = e;  f[-1] = ct_e     # extension accepted
        else:
            vertices[-1] = r;  f[-1] = ct_r     # reflection accepted
    else:
        # contraction: t = (g + worst)/2
        t    = constrain_raw((g + vertices[-1]) / 2.0)
        ct_t = cost_fn(t)
        # also try (g + r)/2
        t2   = constrain_raw((g + r) / 2.0)
        ct_t2 = cost_fn(t2)
        if ct_t2 < ct_t:
            t, ct_t = t2, ct_t2

        if ct_t <= f[-1]:
            vertices[-1] = t;  f[-1] = ct_t     # contraction accepted
        else:
            # shrink — MATLAB: SortVertices(2:n) = (SortVertices(2:n)+SortVertices(1))/2
            vertices[1:] = (vertices[1:] + vertices[0]) / 2.0
            f[1:] = np.array([cost_fn(v) for v in vertices[1:]])

    return vertices, f


# ─────────────────────────────────────────────────────────────────────────────
#  SCE-UA outer loop  (main MATLAB script)
# ─────────────────────────────────────────────────────────────────────────────

def sce_ua(cost_fn,
           n_var: int    = NA,
           P: int        = 12,
           n: int        = 15,
           eps: float    = 1e-4,
           max_iter: int = 500,
           seed: int     = 42,
           verbose: bool = True) -> tuple:
    """
    Shuffled Complex Evolution (SCE-UA).
    Faithful to main MATLAB script (Mkaouar 2021):
      N = 2*n+1 points per complex
      P complexes
      Convergence: sqrt(sum(var(SortVertices))) < eps

    Returns (best_raw_lidf, best_cost).
    """
    np.random.seed(seed)
    N     = 2*n + 1
    total = N * P

    # initialise — MATLAB: rand(N*P, n_variable); normalise; subtract 1/n
    pop = np.random.rand(total, n_var)
    pop = pop / pop.sum(axis=1, keepdims=True) - 1.0/n_var

    if verbose:
        print(f"    SCE-UA: P={P} complexes × N={N} pts = {total} total")

    f_pop = np.array([cost_fn(v) for v in pop])

    for it in range(max_iter):
        conv = float(np.sqrt(np.sum(np.var(pop, axis=0))))
        if conv < eps:
            if verbose:
                print(f"    Converged at iter {it}  conv={conv:.2e}")
            break

        # global sort
        order = np.argsort(f_pop)
        pop   = pop[order];  f_pop = f_pop[order]

        # reshape into P complexes
        R_pop = pop.reshape(P, N, n_var)
        R_f   = f_pop.reshape(P, N)

        for ic in range(P):
            cv = R_pop[ic].copy()
            cf = R_f[ic].copy()

            # select n+1 random points from complex
            sel = np.random.choice(N, n+1, replace=False)
            sv  = cv[sel].copy()
            sf  = cf[sel].copy()

            sv, sf = simplex_step(sv, sf, cost_fn)

            cv[sel]   = sv
            cf[sel]   = sf
            R_pop[ic] = cv
            R_f[ic]   = cf

        pop   = R_pop.reshape(total, n_var)
        f_pop = R_f.reshape(total)

        if verbose and it % 10 == 0:
            print(f"    iter {it:4d} | cost={f_pop.min():.6f} | conv={conv:.4f}")

    best = int(np.argmin(f_pop))
    if verbose:
        print(f"    Final cost = {f_pop[best]:.6f}")
    return pop[best], float(f_pop[best])
