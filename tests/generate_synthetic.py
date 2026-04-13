"""Generate synthetic .npy test dataset with known PAD and LIDF."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from padlad.sail import new_ks_wo, ncampbell

def generate(n_rays=5000, pad=0.4, ala=57, plot_size=20.0,
             canopy_h=15.0, scanner=None, voxel_d=0.5, seed=42,
             outfile=None):
    np.random.seed(seed)
    if scanner is None:
        scanner = np.array([plot_size/2, plot_size/2, 1.3])
    lidf = ncampbell(ala)
    hits = []
    for _ in range(n_rays):
        theta = np.random.uniform(0.05, np.pi/2)
        phi   = np.random.uniform(0, 2*np.pi)
        dx = np.sin(theta)*np.cos(phi)
        dy = np.sin(theta)*np.sin(phi)
        dz = np.cos(theta)
        ks, _ = new_ks_wo(np.degrees(theta), lidf)
        G     = ks * np.cos(theta)
        # Beer-Lambert: expected interception depth
        depth = np.random.exponential(1.0 / (pad * G + 1e-9))
        pt    = scanner + depth * np.array([dx, dy, dz])
        if 0 < pt[2] < canopy_h and 0 < pt[0] < plot_size and 0 < pt[1] < plot_size:
            hits.append(pt)
    pts = np.array(hits)
    meta = {"scanner": scanner.tolist(), "true_pad": pad, "true_ala": ala}
    if outfile:
        np.save(outfile, pts)
        print(f"Saved {len(pts)} points → {outfile}")
    return pts, scanner, meta

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="tests/data/synthetic_plot.npy")
    p.add_argument("--pad",     type=float, default=0.4)
    p.add_argument("--ala",     type=int,   default=57)
    p.add_argument("--n_rays",  type=int,   default=5000)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate(args.n_rays, args.pad, args.ala, outfile=args.output)
