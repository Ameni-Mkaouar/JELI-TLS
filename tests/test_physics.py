"""
tests/test_physics.py
=====================
Physics verification and unit tests for padlad.

Run without pytest:  python tests/test_physics.py
Run with pytest:     python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, shutil

from padlad.sail import (volscatt, new_ks_wo, ncampbell,
                          constrain_lidf, constrain_raw, LITAB, NA)
from padlad.geometry import (sph_tri_area_vos, voxel_solid_angle_areas,
                               woo_traverse, build_grid)
from padlad.inversion import (compute_pad_voxels, cost_variance,
                                simplex_step, sce_ua)
from padlad.pipeline import (compute_transmittance, make_zenith_bins,
                               vertical_profile)
from padlad.io import load_point_cloud

_results = []
def check(name, cond, info=""):
    _results.append((name, cond))
    print(f"  {'v' if cond else 'X'}  {name}" + (f"  [{info}]" if info and not cond else ""))


def test_input_formats():
    print("\n=== 1. Input formats ===")
    import tempfile, os
    pts_ref = np.random.rand(100, 3) * 10
    td = tempfile.mkdtemp()

    np.save(f"{td}/t.npy", pts_ref)
    np.savetxt(f"{td}/t.txt", pts_ref, fmt="%.6f")
    np.savetxt(f"{td}/t.csv", pts_ref, fmt="%.6f", delimiter=",")
    np.savetxt(f"{td}/t_tab.txt", pts_ref, fmt="%.6f", delimiter="\t")
    with open(f"{td}/t_hdr.txt", "w") as f:
        f.write("# X Y Z\n"); np.savetxt(f, pts_ref, fmt="%.6f")
    np.savetxt(f"{td}/t_ext.txt",
               np.hstack([pts_ref, np.random.rand(100,4)]), fmt="%.6f")
    shutil.copy(f"{td}/t.txt", f"{td}/t.xyz")
    shutil.copy(f"{td}/t_hdr.txt", f"{td}/t.pts")

    for ext, label in [(".npy",".npy"), (".txt",".txt space"),
                        (".csv",".csv comma"), ("_tab.txt",".txt tab"),
                        ("_hdr.txt",".txt header"), ("_ext.txt",".txt 7-col"),
                        (".xyz",".xyz"), (".pts",".pts")]:
        p, _ = load_point_cloud(f"{td}/t{ext}")
        check(label, p.shape == (100,3) and np.allclose(p, pts_ref, atol=1e-5), p.shape)

    shutil.rmtree(td)


def test_volscatt():
    print("\n=== 2. volscatt (Verhoef SAIL) ===")
    cs, co, _, _ = volscatt(0, 0, 0, 0)
    check("chi_s = 1 at nadir, horizontal leaf", abs(cs-1)<1e-5, cs)
    check("chi_o = 1 at nadir, horizontal leaf", abs(co-1)<1e-5, co)
    cs90, _, _, _ = volscatt(89.9, 0, 0, 0)
    check("chi_s approx 0 at horizon", abs(cs90)<0.05, cs90)
    check("frho >= 0 for all combinations",
          all(volscatt(t,0,0,l)[2]>=0 for t in [0,20,30,57,75] for l in [5,25,45,75,89]))
    check("ftau >= 0 for all combinations (MATLAB clip)",
          all(volscatt(t,0,0,l)[3]>=0 for t in [0,20,30,57,75] for l in [5,25,45,75,89]))


def test_g_function():
    print("\n=== 3. G-function (new_ks_wo) ===")
    Gs = [new_ks_wo(t, ncampbell(57))[0]*np.cos(np.radians(t)) for t in [10,30,57,75]]
    check("G approx 0.5 for spherical LIDF at all angles",
          all(0.3<g<0.7 for g in Gs), [round(g,3) for g in Gs])
    check("G range < 0.3 for spherical LIDF", max(Gs)-min(Gs)<0.3)
    check("Planophile: G(nadir) > G(horizon)",
          new_ks_wo(10,ncampbell(10))[0]*np.cos(np.radians(10)) >
          new_ks_wo(80,ncampbell(10))[0]*np.cos(np.radians(80)))
    check("Erectophile: G(horizon) > G(nadir)",
          new_ks_wo(80,ncampbell(80))[0]*np.cos(np.radians(80)) >
          new_ks_wo(10,ncampbell(80))[0]*np.cos(np.radians(10)))


def test_ncampbell():
    print("\n=== 4. ncampbell (Campbell 1986) ===")
    for ala in [10, 30, 57, 75, 85]:
        lidf = ncampbell(ala)
        check(f"sum = 1  ALA={ala}", abs(lidf.sum()-1.0)<1e-9)
        check(f"all >= 0 ALA={ala}", lidf.min()>=0)
    check("Planophile peaks low angles", ncampbell(10)[:5].sum()>ncampbell(10)[10:].sum())
    check("Erectophile peaks high angles", ncampbell(80)[10:].sum()>ncampbell(80)[:5].sum())


def test_constrain_lidf():
    print("\n=== 5. constrain_lidf (simplex_tst.m) ===")
    np.random.seed(42)
    for i in range(6):
        raw = np.random.randn(NA)*0.5; c = constrain_lidf(raw)
        check(f"sum = 1 trial {i}", abs(c.sum()-1.0)<1e-9)
        check(f"min >= 0 trial {i}", c.min()>=0)
    check("Extreme negatives handled", constrain_lidf(-np.ones(NA)).min()>=0)
    raw = np.random.randn(NA)*0.1
    check("Round-trip constrain_lidf(constrain_raw(x)) == constrain_lidf(x)",
          np.allclose(constrain_lidf(raw), constrain_lidf(constrain_raw(raw)), atol=1e-9))


def test_solid_angle():
    print("\n=== 6. Solid angle (VOS formula) ===")
    oct_sa = sph_tri_area_vos(np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.]))
    check("Octant solid angle = pi/2", abs(oct_sa-np.pi/2)<0.01, round(oct_sa,4))
    sc = np.zeros(3)
    sa_c = voxel_solid_angle_areas(np.array([[1.,1.,1.]]), sc, 0.5)[0]
    sa_m = voxel_solid_angle_areas(np.array([[5.,5.,5.]]), sc, 0.5)[0]
    sa_f = voxel_solid_angle_areas(np.array([[10.,10.,10.]]), sc, 0.5)[0]
    check("SA decreases with distance: close > mid > far", sa_c>sa_m>sa_f,
          f"{sa_c:.5f}>{sa_m:.5f}>{sa_f:.5f}")
    sa1 = voxel_solid_angle_areas(np.array([[10.,0.,0.]]), sc, 0.1)[0]
    sa2 = voxel_solid_angle_areas(np.array([[20.,0.,0.]]), sc, 0.1)[0]
    check("SA proportional to 1/r2 (ratio approx 4)", abs(sa1/sa2-4.0)<0.5, round(sa1/sa2,2))


def test_woo_traverse():
    print("\n=== 7. woo_traverse (woorayTrace.m) ===")
    vox, d = woo_traverse(np.array([0.25,0.25,0.1]), np.array([0.25,0.25,3.9]),
                           np.zeros(3), 1.0, (4,4,5))
    check("Vertical ray visits z = 0,1,2,3", [v[2] for v in vox]==[0,1,2,3], [v[2] for v in vox])
    check("Distances monotonically increasing", all(d1<=d2 for d1,d2 in zip(d,d[1:])))
    vox2, _ = woo_traverse(np.array([0.1,0.5,0.5]), np.array([3.9,0.5,0.5]),
                            np.zeros(3), 1.0, (5,5,5))
    check("Horizontal ray visits x = 0,1,2,3", [v[0] for v in vox2]==[0,1,2,3])
    vox3, d3 = woo_traverse(np.array([0.1,0.1,0.1]), np.array([3.9,3.9,3.9]),
                             np.zeros(3), 1.0, (5,5,5))
    check("Diagonal ray visits >= 3 voxels", len(vox3)>=3, len(vox3))
    check("No duplicate voxels", len(vox3)==len(set(vox3)))
    check("Zero-length ray returns empty",
          woo_traverse(np.array([1.,1.,1.]),np.array([1.,1.,1.]),np.zeros(3),1.0,(5,5,5))[0]==[])
    check("Ray outside grid returns empty",
          woo_traverse(np.array([10.,10.,10.]),np.array([20.,20.,20.]),np.zeros(3),1.0,(5,5,5))[0]==[])
    vox5, _ = woo_traverse(np.array([10.25,10.25,0.1]), np.array([10.25,10.25,3.9]),
                            np.array([10.,10.,0.]), 1.0, (4,4,5))
    check("Offset-origin vertical ray visits z = 0,1,2,3",
          [v[2] for v in vox5]==[0,1,2,3], [v[2] for v in vox5])


def test_beam_divergence():
    print("\n=== 8. Beam divergence correction (VarLAI_corr2.m) ===")
    rl = 1.0; gs = 0.5
    cor0 = (rl**2*gs)/((rl+0)*(rl*gs+0))
    check("Correction = 1 at scanner position (lidar_tab=0)", abs(cor0-1.0)<1e-9)
    cors = [(rl**2*gs)/((rl+t)*(rl*gs+t)) for t in [0,1,5,20,100]]
    check("Correction decreases monotonically with distance",
          all(c1>=c2 for c1,c2 in zip(cors,cors[1:])))
    np.random.seed(42); N = 15
    cv  = np.random.uniform(1,8,(N,3)); Tr = np.random.uniform(0.3,0.9,N)
    dL  = np.ones(N)*0.5; lidf = np.ones(NA)/NA
    xth = 45*np.ones(N); com = np.cos(np.radians(45))*np.ones(N)
    p1 = compute_pad_voxels(Tr,xth,com,dL,lidf,cv,np.zeros(3),0.0)
    p2 = compute_pad_voxels(Tr,xth,com,dL,lidf,cv,np.zeros(3),7e-5,1.0)
    check("PAD with and without correction differ", not np.allclose(p1,p2))
    check("PAD > 0 (both modes)", np.all(p1>0) and np.all(p2>0))


def test_pvlad():
    print("\n=== 9. PVlad physics (Yin, Cook & Morton 2022) ===")
    np.random.seed(7)
    scanner = np.array([5.,5.,0.])
    pts = np.random.uniform(1,9,(400,3)); pts[:,2] = np.abs(pts[:,2]-5)+0.5
    origin, gs, _ = build_grid(pts, 1.0)

    cv_pv, Tr_pv, dL_pv, _, _, _, val_pv = compute_transmittance(
        pts, scanner, origin, gs, 1.0, set(), subsample=300,
        use_pvlad=True, pe_threshold=0.5)
    check("PVlad ON: valid voxels exist", val_pv.sum()>5, val_pv.sum())
    check("PVlad ON: Tr in (0,1)", np.all(Tr_pv[val_pv]>0)&np.all(Tr_pv[val_pv]<1))
    check("PVlad ON: deltaL > 0", np.all(dL_pv[val_pv]>0))

    _, Tr_mk, _, _, _, _, val_mk = compute_transmittance(
        pts, scanner, origin, gs, 1.0, set(), subsample=300,
        use_pvlad=False, min_beams=3)
    check("Original ON: valid voxels exist", val_mk.sum()>5, val_mk.sum())

    common = val_pv & val_mk
    if common.sum() >= 5:
        diff = float(np.max(np.abs(Tr_pv[common]-Tr_mk[common])))
        check("Range weighting changes Tr values (PVlad vs standard)",
              diff>1e-6, f"max_diff={diff:.6f}")

    _, _, _, _, _, _, val_strict = compute_transmittance(
        pts, scanner, origin, gs, 1.0, set(), subsample=300,
        use_pvlad=True, pe_threshold=10.0)
    _, _, _, _, _, _, val_loose = compute_transmittance(
        pts, scanner, origin, gs, 1.0, set(), subsample=300,
        use_pvlad=True, pe_threshold=0.1)
    check("Higher PE threshold -> fewer valid voxels",
          val_strict.sum() <= val_loose.sum(),
          f"strict={val_strict.sum()} loose={val_loose.sum()}")


def test_sce_ua():
    print("\n=== 10. SCE-UA optimiser ===")
    np.random.seed(42)
    verts = np.random.randn(5,NA)*0.1; costs = np.random.rand(5)
    def cf(raw): return float(np.var(np.abs(raw+1.0/NA)))
    fb = costs.min(); _, f_new = simplex_step(verts, costs, cf)
    check("Simplex step does not worsen best cost", f_new.min()<=fb+1e-6)

    np.random.seed(123)
    sc3 = np.zeros(3); tl = ncampbell(57); N3 = 60
    cv3 = np.random.uniform(1,8,(N3,3))
    diff3 = cv3-sc3; dist3 = np.linalg.norm(diff3,axis=1)
    cos3 = np.abs(diff3[:,2])/(dist3+1e-12)
    xth3 = np.degrees(np.arccos(np.clip(cos3,0,1)))
    dL3 = np.ones(N3)*0.5
    Tr3 = np.clip(np.array([np.exp(-0.4*new_ks_wo(xth3[i],tl)[0]*cos3[i]*dL3[i])
                             for i in range(N3)])+np.random.normal(0,0.02,N3),0.05,0.95)
    vr = [v for v in [np.where((xth3>=i)&(xth3<i+15))[0] for i in range(0,90,15)] if len(v)>=2]
    def c2(raw): return cost_variance(raw,Tr3,xth3,cos3,dL3,vr,cv3,sc3,0.0,1.0)
    best_raw, _ = sce_ua(c2, n_var=NA, P=4, n=5, max_iter=60, verbose=False)
    est = constrain_lidf(best_raw)
    check("SCE-UA recovers spherical LIDF rather than planophile",
          np.sum((est-ncampbell(57))**2) < np.sum((est-ncampbell(10))**2))


def test_integration():
    print("\n=== 11. Integration — PVlad ON and OFF ===")
    np.random.seed(7)
    scanner = np.array([5.,5.,0.]); tl = ncampbell(57)
    pts = []
    for _ in range(3000):
        theta = np.random.uniform(0.05,np.pi/2); phi = np.random.uniform(0,2*np.pi)
        ks, _ = new_ks_wo(np.degrees(theta), tl); G = ks*np.cos(theta)
        depth = np.random.exponential(1.0/(0.4*G+1e-9))
        pt = scanner + depth*np.array([np.sin(theta)*np.cos(phi),
                                        np.sin(theta)*np.sin(phi), np.cos(theta)])
        if 0<pt[2]<12 and 0<pt[0]<10 and 0<pt[1]<10: pts.append(pt)
    pts = np.array(pts[:800])
    origin, gs, _ = build_grid(pts, 1.0)

    for mode, use_pv, pe_thr, mb in [("PVlad ON",True,0.5,3), ("PVlad OFF",False,0,3)]:
        if use_pv:
            cv4,Tr4,dL4,xth4,co4,_,val4 = compute_transmittance(
                pts,scanner,origin,gs,1.0,set(),subsample=400,
                use_pvlad=True, pe_threshold=pe_thr)
        else:
            cv4,Tr4,dL4,xth4,co4,_,val4 = compute_transmittance(
                pts,scanner,origin,gs,1.0,set(),subsample=400,
                use_pvlad=False, min_beams=mb)
        check(f"{mode}: valid voxels", val4.sum()>5)
        vr4, _, nz4 = make_zenith_bins(xth4, val4, 10.0)
        check(f"{mode}: zenith bins", nz4>0)
        vi4 = np.where(val4)[0]; pm = {g:l for l,g in enumerate(vi4)}
        vr4l = [v for v in [np.array([pm[g] for g in grp if g in pm]) for grp in vr4] if len(v)>0]
        def c3(raw): return cost_variance(raw,Tr4[vi4],xth4[vi4],co4[vi4],dL4[vi4],vr4l,cv4[vi4],scanner,0.0,1.0)
        br2, _ = sce_ua(c3, n_var=NA, P=3, n=4, max_iter=15, verbose=False)
        el2 = constrain_lidf(br2)
        check(f"{mode}: LIDF sums to 1", abs(el2.sum()-1.0)<1e-9)
        df4 = vertical_profile(cv4,Tr4,dL4,xth4,co4,scanner,br2,1.0,0.0,1.0)
        check(f"{mode}: profile has rows", len(df4)>0)
        check(f"{mode}: PAD > 0", (df4["PAD"]>0).all())
        check(f"{mode}: LAI in range", 0.01<df4["LAI_total"].iloc[0]<20,
              f"LAI={df4['LAI_total'].iloc[0]:.3f}")


def run_all():
    print("="*60); print("  padlad — physics and unit tests"); print("="*60)
    test_input_formats()
    test_volscatt()
    test_g_function()
    test_ncampbell()
    test_constrain_lidf()
    test_solid_angle()
    test_woo_traverse()
    test_beam_divergence()
    test_pvlad()
    test_sce_ua()
    test_integration()
    passed = sum(1 for _,c in _results if c)
    failed = sum(1 for _,c in _results if not c)
    print(f"\n{'='*60}")
    print(f"  PASSED: {passed}/{passed+failed}")
    print(f"  FAILED: {failed}")
    if failed:
        print("\n  Failed tests:")
        for name,ok in _results:
            if not ok: print(f"    X  {name}")
    print("="*60)
    return failed==0

def test_all_physics():
    """pytest entry point."""
    assert run_all()

if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
