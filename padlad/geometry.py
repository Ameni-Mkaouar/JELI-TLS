"""
padlad/geometry.py
==================
Voxel geometry, solid angle computation, and Woo (1987) ray traversal.
Faithful translation of VoxelSolidAngleArea.m and woorayTrace.m.

Enhancement: replaces unstable dihedral-angle formula with
Van Oosterom & Strackee (1983) for spherical triangle area —
numerically stable and physically correct for all voxel distances.
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Spherical triangle area — Van Oosterom & Strackee (1983)
#  Enhancement over MATLAB dihedral formula: numerically stable at all ranges
# ─────────────────────────────────────────────────────────────────────────────

def sph_tri_area_vos(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Solid angle of triangle (a, b, c) in steradians.
    Uses Van Oosterom & Strackee (1983):
        Ω = 2 arctan(|a·(b×c)| / (1 + a·b + b·c + a·c))

    Numerically stable for all distances — replaces the dihedral-angle
    method used in MATLAB (which diverges for close voxels).
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    nc = np.linalg.norm(c)
    if na < 1e-15 or nb < 1e-15 or nc < 1e-15:
        return 0.0
    a = a / na;  b = b / nb;  c = c / nc
    num = abs(float(np.dot(a, np.cross(b, c))))
    den = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(a, c))
    if abs(den) < 1e-15:
        return 0.0
    return 2.0 * np.arctan2(num, den)


def voxel_solid_angle_areas(centre_vox: np.ndarray,
                             scanner: np.ndarray,
                             voxel_d: float) -> np.ndarray:
    """
    Solid angle subtended by each voxel at the scanner position.
    Uses three face triangles per voxel — faithful to VoxelSolidAngleArea.m
    structure (triangle1, triangle2, triangle3 × 2).

    Enhancement: uses VOS formula instead of MATLAB's SphTriAreaNorm
    (dihedral-angle method) which is numerically unstable for near voxels.

    Parameters
    ----------
    centre_vox : (N, 3) voxel centres
    scanner    : (3,)   scanner position
    voxel_d    : voxel edge length

    Returns
    -------
    angle_area : (N,) solid angles in steradians, monotonically
                 decreasing with distance (physically correct)
    """
    N   = len(centre_vox)
    out = np.zeros(N)
    d   = voxel_d
    dx  = np.array([d, 0.0, 0.0])
    dy  = np.array([0.0, d, 0.0])
    dz  = np.array([0.0, 0.0, d])

    # lower corner of each voxel (x0y0z0 in MATLAB)
    p0 = centre_vox - d / 2.0

    for i in range(N):
        o = p0[i] - scanner               # translate to scanner origin
        # triangle1: xy-face
        s1 = sph_tri_area_vos(o, o+dx, o+dy)
        # triangle2: xz-face
        s2 = sph_tri_area_vos(o, o+dx, o+dz)
        # triangle3: yz-face
        s3 = sph_tri_area_vos(o, o+dy, o+dz)
        out[i] = (s1 + s2 + s3) * 2.0    # ×2 matches MATLAB

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Woo (1987) ray traversal — woorayTrace.m
#  Enhancement: vectorised t-parameter init, explicit boundary handling
# ─────────────────────────────────────────────────────────────────────────────

def woo_traverse(scanner: np.ndarray,
                 end_pt: np.ndarray,
                 origin: np.ndarray,
                 vsize: float,
                 shape: tuple) -> tuple:
    """
    Amanatides & Woo (1987) fast voxel traversal.
    Faithful translation of woorayTrace.m (Ameni Mkaouar).

    Key: MATLAB uses parametric t ∈ [0,1] where t=1 means end_pt.
    lineCoord = [scanner_x, scanner_y, scanner_z, end_x, end_y, end_z].

    Returns (voxel_0based_indices_list, relative_distances_list).
    relative_distances match MATLAB Distance = calc_dist(tMaxX,tMaxY,tMaxZ).
    """
    # Full direction vector (not normalised) — matches MATLAB xVec/yVec/zVec
    xVec = end_pt[0] - scanner[0]
    yVec = end_pt[1] - scanner[1]
    zVec = end_pt[2] - scanner[2]

    if abs(xVec) < 1e-12 and abs(yVec) < 1e-12 and abs(zVec) < 1e-12:
        return [], []

    # Grid bounds
    gx0, gy0, gz0 = origin
    gx1 = origin[0] + shape[0] * vsize
    gy1 = origin[1] + shape[1] * vsize
    gz1 = origin[2] + shape[2] * vsize

    # Parametric intersection with grid slab — matching MATLAB divX logic
    # t=0 → scanner, t=1 → end_pt
    def _slab(vec, lo, hi, p0):
        if abs(vec) < 1e-12:
            return -np.inf, np.inf
        t1 = (lo - p0) / vec
        t2 = (hi - p0) / vec
        return (t1, t2) if vec > 0 else (t2, t1)

    txMin, txMax = _slab(xVec, gx0, gx1, scanner[0])
    tyMin, tyMax = _slab(yVec, gy0, gy1, scanner[1])
    tzMin, tzMax = _slab(zVec, gz0, gz1, scanner[2])

    tMin = max(txMin, tyMin, tzMin, 0.0)
    tMax = min(txMax, tyMax, tzMax, 1.0)   # MATLAB: tMax = min(tMax, 1)

    if tMin >= tMax:
        return [], []

    # voxel sizes (uniform here, matching MATLAB voxelSizeX/Y/Z)
    vSX = vsize;  vSY = vsize;  vSZ = vsize

    # Entry and exit positions
    xStart = scanner[0] + xVec * tMin
    yStart = scanner[1] + yVec * tMin
    zStart = scanner[2] + zVec * tMin
    xEnd   = scanner[0] + xVec * tMax
    yEnd   = scanner[1] + yVec * tMax
    zEnd   = scanner[2] + zVec * tMax

    # Starting voxel — MATLAB: X = max(1, ceil((xStart-gridBounds(1))/voxelSizeX))
    # Convert to 0-based: X0 = X_matlab - 1
    def _start_idx(pos, lo, vs, sz):
        return max(0, min(int(np.ceil((pos - lo) / vs)) - 1, sz - 1))

    def _end_idx(pos, lo, vs, sz):
        return max(0, min(int(np.ceil((pos - lo) / vs)) - 1, sz - 1))

    X = _start_idx(xStart, gx0, vSX, shape[0])
    Y = _start_idx(yStart, gy0, vSY, shape[1])
    Z = _start_idx(zStart, gz0, vSZ, shape[2])
    Xend = _end_idx(xEnd, gx0, vSX, shape[0])
    Yend = _end_idx(yEnd, gy0, vSY, shape[1])
    Zend = _end_idx(zEnd, gz0, vSZ, shape[2])

    # Step direction and parametric delta per dimension
    # MATLAB: tMaxX = tMin + (gridBounds(1) + X*voxelSizeX - xStart) / xVec
    # where X is 1-based → (X*vSX) is the right boundary of voxel X
    # In 0-based: right boundary of voxel X0 = gx0 + (X0+1)*vSX
    if xVec > 1e-12:
        stepX   = 1
        tDeltaX = vSX / xVec
        tMaxX   = tMin + (gx0 + (X+1)*vSX - xStart) / xVec
    elif xVec < -1e-12:
        stepX   = -1
        tDeltaX = vSX / (-xVec)
        tMaxX   = tMin + (gx0 + X*vSX - xStart) / xVec
    else:
        stepX   = 0
        tDeltaX = tMax
        tMaxX   = tMax

    if yVec > 1e-12:
        stepY   = 1
        tDeltaY = vSY / yVec
        tMaxY   = tMin + (gy0 + (Y+1)*vSY - yStart) / yVec
    elif yVec < -1e-12:
        stepY   = -1
        tDeltaY = vSY / (-yVec)
        tMaxY   = tMin + (gy0 + Y*vSY - yStart) / yVec
    else:
        stepY   = 0
        tDeltaY = tMax
        tMaxY   = tMax

    if zVec > 1e-12:
        stepZ   = 1
        tDeltaZ = vSZ / zVec
        tMaxZ   = tMin + (gz0 + (Z+1)*vSZ - zStart) / zVec
    elif zVec < -1e-12:
        stepZ   = -1
        tDeltaZ = vSZ / (-zVec)
        tMaxZ   = tMin + (gz0 + Z*vSZ - zStart) / zVec
    else:
        stepZ   = 0
        tDeltaZ = tMax
        tMaxZ   = tMax

    traversed = []
    rel_dists = []
    max_steps = abs(X - Xend) + abs(Y - Yend) + abs(Z - Zend) + 2

    for _ in range(max_steps + 1):
        if (X < 0 or X >= shape[0] or
                Y < 0 or Y >= shape[1] or
                Z < 0 or Z >= shape[2]):
            break

        traversed.append((X, Y, Z))
        # calc_dist.m: dist = min(tMaxX, tMaxY, tMaxZ)
        rel_dists.append(float(min(tMaxX, tMaxY, tMaxZ)))

        if X == Xend and Y == Yend and Z == Zend:
            break

        # Advance to next boundary — MATLAB stepping logic
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                X    += stepX
                tMaxX += tDeltaX
            else:
                Z    += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                Y    += stepY
                tMaxY += tDeltaY
            else:
                Z    += stepZ
                tMaxZ += tDeltaZ

    return traversed, rel_dists


# ─────────────────────────────────────────────────────────────────────────────
#  Grid utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(pts: np.ndarray, voxel_d: float) -> tuple:
    """
    Build voxel grid bounds — matches MATLAB:
      xmin = floor(min/d)*d - 2d
      xmax = floor(max/d)*d + 2d
    Returns (origin, grid_size, bounds[6]).
    """
    lo = np.floor(pts.min(axis=0) / voxel_d) * voxel_d - 2*voxel_d
    hi = np.floor(pts.max(axis=0) / voxel_d) * voxel_d + 2*voxel_d
    grid_size = np.floor((hi - lo) / voxel_d).astype(int)
    return lo, grid_size, np.concatenate([lo, hi])


def tot_pt_voxel(pas: float,
                 xth: np.ndarray,
                 angle_area: np.ndarray,
                 corsp_indx: np.ndarray,
                 tot_pt_ang: np.ndarray,
                 lv2: int) -> np.ndarray:
    """
    Estimate total rays per voxel using solid angle weighting.
    Faithful translation of TotPtVoxel.m.

    pt_ang(indx) = tot_pt_ang(j) * Angle_area(indx)
    """
    pt_ang = np.zeros(lv2)
    j = 0
    i = 0.0
    while i <= 90 - pas:
        indx = np.where((xth >= i) & (xth < i + pas))[0]
        if len(indx) > 0 and j < len(tot_pt_ang):
            pt_ang[indx] = tot_pt_ang[j] * angle_area[indx]
            j += 1
        i += pas
    return pt_ang[corsp_indx]
