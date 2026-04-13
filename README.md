# padlad-tls

**Joint PAD/LAD Estimator from TLS Point Clouds**

Python implementation of the method described in:

> **[1]** Mkaouar, A., Kallel, A., Guidara, R., Ben Rabah, Z., Sahli, T., Qi, J., & Gastellu-Etchegorry, J.-P. (2021). *Joint Estimation of Leaf Area Density and Leaf Angle Distribution Using TLS Point Cloud for Forest Stands.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 14, 11095–11115. https://doi.org/10.1109/JSTARS.2021.3120521

> **[2]** Mkaouar, A., & Kallel, A. (2023). *Leaf properties estimation enhancement over heterogeneous vegetation by correcting for terrestrial laser scanning beam divergence effect.* Remote Sensing of Environment. https://doi.org/10.1016/j.rse.2023.11395 

---

## What makes this different

Most PAD tools (AMAPVox, L-Vox, lidR) assume a **fixed G = 0.5** (spherical leaf distribution). This tool **jointly estimates PAD and LIDF simultaneously** — the leaf angle distribution is an output, not an assumption.

The key insight: if the LIDF estimate is wrong, PAD will be *inconsistent across viewing directions*. The SCE-UA optimiser finds the LIDF that makes PAD maximally consistent.

Paper [2] adds a **beam divergence correction**: the laser footprint grows with distance from the scanner, biasing transmittance estimates for distant voxels.

---

## Installation

```bash
git clone https://github.com/your-username/padlad-tls
cd padlad-tls
pip install -r requirements.txt
```

**Requirements:** `numpy scipy matplotlib pandas laspy[lazrs]`

---

## Quick start

### Single scan — Paper [1] (no beam divergence correction)
```bash
python cli.py \
    --input plot.laz \
    --voxel_size 0.5 \
    --scanner_x 19.93 --scanner_y 50.07 --scanner_z 0.001 \
    --output results/
```

### Single scan — Paper [2] (with beam divergence correction)
```bash
python cli.py \
    --input plot.laz \
    --voxel_size 0.5 \
    --scanner_x 19.93 --scanner_y 50.07 --scanner_z 0.001 \
    --beam_divergence 7e-5 \
    --output results/
```

### Multi-scan plot 
Process each scan position independently, then merge:
```bash
# Each scan with its own scanner position
python cli.py --input scan1.laz --scanner_x 0   --scanner_y 0   --scanner_z 1.3 --output results/scan1/
python cli.py --input scan2.laz --scanner_x 10  --scanner_y 0   --scanner_z 1.3 --output results/scan2/
python cli.py --input scan3.laz --scanner_x 10  --scanner_y 10  --scanner_z 1.3 --output results/scan3/
python cli.py --input scan4.laz --scanner_x 0   --scanner_y 10  --scanner_z 1.3 --output results/scan4/

# Merge profiles (weighted by n_voxels per height layer)
python cli.py --merge results/scan*/scan*_PAD_profile.csv --output results/merged_profile.csv
```

⚠️ **Important**: Do NOT merge point clouds before running. Each scan must be processed with its own scanner position. The physics (solid angle areas, distance-decay curves) depend on a single fixed origin. See [multi-scan note](#multi-scan-note) below.

### Python API
```python
import numpy as np
from padlad.io import load_point_cloud, get_scanner_position, remove_ground, save_outputs
from padlad.geometry import build_grid
from padlad.pipeline import run_pipeline

pts, meta   = load_point_cloud("plot.laz")
scanner     = get_scanner_position(meta, pts, manual=[19.93, 50.07, 0.001])
pts, _      = remove_ground(pts)

df, lidf, _ = run_pipeline(
    pts, scanner,
    voxel_d      = 0.5,
    beam_div     = 0.0,     # or 7e-5 for Paper [2]
    sce_maxiter  = 300,
)

print(f"LAI = {df['LAI_total'].iloc[0]:.3f}")
print(f"Peak PAD = {df['PAD'].max():.3f} at z = {df.loc[df['PAD'].idxmax(),'height_m']:.2f} m")
```

---

## Method

### Pipeline

| Step | Source | Function |
|------|--------|----------|
| Grid bounds | Main script | `build_grid()` |
| Woo ray traversal | `woorayTrace.m` | `woo_traverse()` |
| Transmittance per voxel | Main script | `compute_transmittance()` |
| Solid angle per voxel | `VoxelSolidAngleArea.m` | `voxel_solid_angle_areas()` |
| SAIL G-function | `new_ks_wo2.m` | `new_ks_wo()` |
| SCE-UA simplex step | `simplex_tst.m` | `simplex_step()` |
| SCE-UA outer loop | Main script | `sce_ua()` |
| Cost function | `VarLAI_corr2.m` | `cost_variance()` |
| Beam div. correction | `VarLAI_corr2.m` | `compute_pad_voxels()` |
| Campbell LIDF init | `ncampbell.m` | `ncampbell()` |

### LIDF parameterisation

The 15-bin LIDF uses angle midpoints from `litab = [5,15,25,35,45,55,64,72,77,79,81,83,85,87,89]` degrees, matching the original MATLAB scripts exactly.

### Cost function

`cost = var(mean_PAD per 2° zenith bin)`

Minimising this finds the LIDF that makes PAD estimates **consistent across all viewing directions** — physically, if the leaf angle distribution is correct, transmittance should predict the same PAD regardless of the viewing angle.

### Beam divergence correction (Paper [2])

```
lidar_tab = tan(ε/2) × ‖voxel_centre − scanner‖
cor = (rl² × G) / ((rl + lidar_tab)(rl × G + lidar_tab))
PAD_corrected = PAD × cor
```

This accounts for the fact that the laser footprint grows with distance, causing far voxels to be sampled by a wider beam and overestimating transmittance.

### Enhancements over original MATLAB

| Issue | Original MATLAB | This Python version |
|-------|----------------|---------------------|
| Solid angle formula | Dihedral angles (numerically unstable for close voxels) | Van Oosterom & Strackee (1983) — stable at all distances |
| `frho`/`ftau` clip | `if frho<0: frho=0` | Faithfully implemented |
| GPU dependency | `gpuArray` for solid angle | Pure NumPy — runs on CPU |

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | `.las`, `.laz`, `.txt`, or `.npy` |
| `--output` | `results/` | Output directory |
| `--voxel_size` | `0.5` | Voxel edge length (m). Paper used 0.5 m. |
| `--scanner_x/y/z` | auto | Scanner position. |
| `--beam_divergence` | `0` | Beam half-angle (rad). `0`=Paper [1]. `7e-5`=Paper [2] birch. |
| `--rl` | `1.0` | Range parameter for beam div. correction. |
| `--stem_cloud` | None | `.txt` stem points for trunk masking. |
| `--subsample` | `10000` | Rays to trace. Increase for accuracy. |
| `--min_beams` | `5` | Minimum rays per voxel. |
| `--pas_zenith` | `2.0` | Zenith bin width (degrees). |
| `--sce_P` | `12` | SCE-UA complexes. |
| `--sce_n` | `15` | SCE-UA simplex parameter. |
| `--sce_eps` | `1e-4` | SCE-UA convergence threshold. |

### Beam divergence values by scanner

| Scanner | ε (half-angle rad) |
|---|---|
| RIEGL VZ-400 / VZ-400i | 3.5e-4 |
| Leica ScanStation P40 | 2.3e-4 |
| Faro Focus 3D | 1.9e-4 |
| Paper [2] dataset (birch) | 7e-5 |

---

## Output files

| File | Contents |
|---|---|
| `{stem}_PAD_profile.csv` | height_m, PAD, n_voxels, LAI_cumulative, LAI_total |
| `{stem}_LIDF.csv` | angle_deg (15 bins), LIDF frequency |
| `{stem}_PAD_profile.png` | Three-panel: PAD profile / LIDF / cumulative LAI |

---

## Multi-scan note

The method is designed for **a single scanner position**. All solid angle computations, distance-decay curve fitting, and the directional pre-interception correction assume rays originate from one fixed point.

For plots with multiple scan positions (e.g. 4–9 positions in a forest inventory):
1. Process each `.laz` file separately with its scanner's known position
2. Use `python cli.py --merge` to combine the resulting PAD profile CSVs
3. Profiles are merged using weighted averaging (weight = n_voxels per height layer), giving more trust to the scan that sampled each layer best

This is not a limitation — it is physically correct, since each scan has its own geometry.

---

## Running tests

```bash
# With pytest:
python -m pytest tests/ -v

# Without pytest:
python tests/test_physics.py
```

Tests cover: `volscatt` physics, G-function patterns, LIDF properties, solid angle inverse-square law, ray traversal geometry, beam divergence correction, and SCE-UA convergence.

---

## Citation

If you use this tool, please cite:

```bibtex
@article{mkaouar2021joint,
  title={Joint Estimation of Leaf Area Density and Leaf Angle Distribution
         Using TLS Point Cloud for Forest Stands},
  author={Mkaouar, Ameni and Kallel, Abdelaziz and others},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations
           and Remote Sensing},
  volume={14},
  pages={11095--11115},
  year={2021},
  doi={10.1109/JSTARS.2021.3120521}
}
```

For the beam divergence correction (Paper [2]):
```bibtex
@article{mkaouar2023beam,
  title={Leaf properties estimation enhancement over heterogeneous vegetation
         by correcting for terrestrial laser scanning beam divergence effect},
  author={Mkaouar, Ameni and Kallel, Abdelaziz},
  journal={Remote Sensing of Environment},
  year={2023}
}
```
