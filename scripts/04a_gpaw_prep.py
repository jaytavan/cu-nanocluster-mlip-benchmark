"""
04a_gpaw_prep.py

GPAW Preparation — MACE H Relaxation & Site Filtering
-------------------------------------------------------
Run this BEFORE 04b_gpaw_launch.py. It does all GPU work in one shot so
the GPAW workers (04b) run CPU-only with no GPU race conditions.

For each representative cluster (Cu10_00 … Cu50_00):
  1. Load MACE-relaxed cluster geometry
  2. Filter sites: |ΔGH*_MACE| < DGH_FILTER (default 0.5 eV)
  3. For each passing site, re-relax H with MACE (frozen cluster) to get
     the optimal H position for the GPAW single-point
  4. Save prepared geometry (cluster+H) and all metadata to:
         results/gpaw/sites_prepared.json   ← coordinates + MACE ΔGH*
         results/gpaw/sites_prepared.traj   ← ASE trajectory (all cluster+H)

Also computes and caches E(H2) with GPAW so workers don't repeat it:
         results/gpaw/h2_energy_gpaw.json

Usage:
    python3 04a_gpaw_prep.py                   # full run, |ΔGH|<0.5 filter
    python3 04a_gpaw_prep.py --filter 0.3      # stricter filter
    python3 04a_gpaw_prep.py --test            # 3 sites per size only
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT      = Path(__file__).parent.parent
RESULTS      = PROJECT / "results"
GPAW_DIR     = RESULTS / "gpaw"
GPAW_DIR.mkdir(exist_ok=True)

CLUSTERS_TRAJ = RESULTS / "clusters_relaxed.traj"
SITES_JSON    = RESULTS / "sites_mace.json"
RELAX_LOG     = RESULTS / "relaxation_log.json"
OUT_JSON      = GPAW_DIR / "sites_prepared.json"
OUT_TRAJ      = GPAW_DIR / "sites_prepared.traj"
H2_CACHE      = GPAW_DIR / "h2_energy_gpaw.json"

# ── Parameters ─────────────────────────────────────────────────────────────
SIZES           = [10, 20, 30, 40, 50]
REP_SUFFIX      = "_00"             # representative cluster per size
DEVICE          = "cuda"
MACE_MODEL      = "medium"
H_FMAX          = 0.10              # eV/Å — H relaxation convergence
H_MAX_STEPS     = 150
ECUT            = 350               # eV — GPAW plane-wave cutoff (for H2 only here)
ZPE_CORR        = 0.24              # eV — standard CHE correction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--filter', type=float, default=0.5,
                   help='|ΔGH*_MACE| threshold for site selection (default: 0.5 eV)')
    p.add_argument('--test', action='store_true',
                   help='Process only 3 sites per size (sanity check)')
    p.add_argument('--ncores', type=int, default=4,
                   help='OMP threads for GPAW H2 calculation (default: 4)')
    return p.parse_args()


# ── MACE setup ──────────────────────────────────────────────────────────────
def load_mace():
    import torch
    from mace.calculators import mace_mp

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run environment check first.")
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print(f"  Model: MACE-MP-0 ({MACE_MODEL}), float32")
    calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float32")
    print("  MACE ready.\n")
    return calc


def relax_h_mace(cluster: Atoms, site_pos: np.ndarray, mace_calc,
                 h_dist: float = 1.7) -> Atoms:
    """
    Place H at site_pos + h_dist*outward, relax H only (cluster frozen).
    Returns cluster+H with H at MACE-optimal position.
    """
    center    = cluster.get_positions().mean(axis=0)
    direction = site_pos - center
    norm      = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction /= norm

    h_pos = site_pos + h_dist * direction

    cluster_h = cluster.copy()
    cluster_h += Atoms('H', positions=[h_pos])
    cluster_h.set_pbc(False)
    cluster_h.set_constraint(FixAtoms(indices=list(range(len(cluster)))))
    cluster_h.calc = mace_calc

    opt = LBFGS(cluster_h, logfile=None)
    opt.run(fmax=H_FMAX, steps=H_MAX_STEPS)
    cluster_h.set_constraint()   # clear constraint for GPAW
    return cluster_h


# ── GPAW H2 reference ───────────────────────────────────────────────────────
def compute_h2_gpaw(ncores: int) -> float:
    """
    Compute E(H2) with GPAW PBE/PW350. Cached to h2_energy_gpaw.json.
    This is the reference that ALL GPAW workers must use.
    """
    if H2_CACHE.exists():
        with open(H2_CACHE) as f:
            data = json.load(f)
        e_h2 = data['e_h2_eV']
        print(f"  E(H2) GPAW loaded from cache: {e_h2:.5f} eV")
        return e_h2

    print("  Computing E(H2) with GPAW...")
    os.environ['OMP_NUM_THREADS']      = str(ncores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(ncores)

    from gpaw import GPAW, PW, Mixer

    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.center(vacuum=7.5)
    h2.set_pbc(False)

    calc = GPAW(
        mode=PW(ECUT),
        xc='PBE',
        kpts={'gamma': True},
        occupations={'name': 'fermi-dirac', 'width': 0.2},
        mixer=Mixer(beta=0.05, nmaxold=5, weight=50),
        txt=str(GPAW_DIR / 'h2_ref.txt'),
        convergence={'energy': 1e-4},
        symmetry='off',
    )
    h2.calc = calc
    t0 = time.time()
    e_h2 = h2.get_potential_energy()
    elapsed = time.time() - t0
    bond = h2.get_distance(0, 1)

    print(f"  E(H2) = {e_h2:.5f} eV  (bond: {bond:.3f} Å)  [{elapsed:.0f}s]")
    with open(H2_CACHE, 'w') as f:
        json.dump({'e_h2_eV': float(e_h2), 'bond_A': float(bond),
                   'ecut_eV': ECUT, 'xc': 'PBE', 'time_s': elapsed}, f, indent=2)
    return float(e_h2)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 65)
    print("04a — GPAW Site Preparation (MACE H relaxation)")
    print(f"  ΔGH* filter:    |ΔGH*_MACE| < {args.filter} eV")
    print(f"  Test mode:      {args.test}")
    print(f"  OMP threads:    {args.ncores}  (for GPAW H2 only)")
    print("=" * 65)

    # Load data
    clusters_all = read(str(CLUSTERS_TRAJ), index=':')
    with open(SITES_JSON) as f:
        sites_data = json.load(f)
    with open(RELAX_LOG) as f:
        relax_logs = json.load(f)

    e_cluster_map = {l['config_id']: l['e_after_eV']
                     for l in relax_logs if 'e_after_eV' in l}

    # Index clusters by config_id
    cluster_map = {a.info.get('config_id'): a for a in clusters_all}

    # ── GPAW H2 reference ───────────────────────────────────────────────────
    print("\nStep 1 — GPAW E(H2) reference")
    e_h2_gpaw = compute_h2_gpaw(args.ncores)

    # ── MACE H relaxation for all filtered sites ────────────────────────────
    print("\nStep 2 — MACE H relaxation for filtered sites")
    print("  Loading MACE...")
    mace_calc = load_mace()

    all_prepared = []    # list of dicts for OUT_JSON
    all_h_structs = []  # for OUT_TRAJ

    for sz in SIZES:
        cid = f"Cu{sz}{REP_SUFFIX}"
        if cid not in sites_data:
            print(f"  WARNING: {cid} not found in sites_mace.json — skipping")
            continue

        cluster = cluster_map.get(cid)
        if cluster is None:
            print(f"  WARNING: {cid} not found in clusters_relaxed.traj — skipping")
            continue

        e_cluster = e_cluster_map.get(cid)
        cluster_sites = sites_data[cid]['sites']

        # Filter sites
        filtered = [s for s in cluster_sites
                    if abs(s['dgh_eV']) < args.filter]

        if args.test:
            filtered = filtered[:3]

        print(f"\n  Cu{sz}_00: {len(cluster_sites)} total sites "
              f"→ {len(filtered)} pass |ΔGH*| < {args.filter} eV"
              + (" [TEST: 3 only]" if args.test else ""))

        t0_size = time.time()
        for i, site in enumerate(filtered):
            # site['atoms'] contains Cu atom indices — compute site centroid
            cu_pos = cluster.get_positions()
            atom_indices = site['atoms']
            site_centroid = cu_pos[atom_indices].mean(axis=0)

            t_site = time.time()
            cluster_h = relax_h_mace(cluster, site_centroid, mace_calc)
            elapsed = time.time() - t_site

            h_pos = cluster_h.get_positions()[-1]

            prepared = {
                'site_global_id': f"{cid}_site{i:03d}",
                'config_id':      cid,
                'size':           sz,
                'site_idx':       i,
                'site_type':      site['type'],
                'site_atoms':     site['atoms'],
                'dgh_mace_eV':    site['dgh_eV'],
                'e_cluster_mace': e_cluster,
                'e_h2_gpaw':      e_h2_gpaw,
                'h_pos':          h_pos.tolist(),
                'cluster_pos':    cluster.get_positions().tolist(),
                'cell':           cluster.get_cell().tolist(),
                'zpe_corr':       ZPE_CORR,
                'mace_h_relax_s': round(elapsed, 2),
            }
            all_prepared.append(prepared)
            all_h_structs.append(cluster_h)

            if (i + 1) % 10 == 0 or i == len(filtered) - 1:
                print(f"    [{i+1}/{len(filtered)}] {site['type']:<7} "
                      f"ΔGH*_MACE={site['dgh_eV']:+.3f} eV  "
                      f"H-relax: {elapsed:.1f}s")

        t_size = time.time() - t0_size
        print(f"  Cu{sz} done: {len(filtered)} sites in {t_size:.1f}s")

    # Release GPU memory before GPAW runs
    del mace_calc
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    print("\n  MACE released — GPU memory freed for GPAW workers.")

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump(all_prepared, f, indent=2)
    write(str(OUT_TRAJ), all_h_structs)

    print(f"\nSaved {len(all_prepared)} prepared sites:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_TRAJ}")

    # Summary table
    print(f"\n{'='*65}")
    print("Sites queued for GPAW (per size):")
    print(f"  {'Size':<8} {'Sites':>6}  {'ΔGH* range (MACE)':>20}  {'Est wall (4-par)':>18}")
    print("  " + "-" * 58)

    time_per_calc = {10: 0.7, 20: 2.9, 30: 6.8, 40: 12.6, 50: 20.4}
    total_sites = 0
    total_time  = 0
    for sz in SIZES:
        group = [p for p in all_prepared if p['size'] == sz]
        if not group:
            continue
        dgh_vals = [p['dgh_mace_eV'] for p in group]
        n = len(group)
        t_seq = n * time_per_calc[sz]
        t_par = t_seq / 4   # estimate with 4 parallel workers
        total_sites += n
        total_time  += t_seq
        def fmt(m):
            return f'{m:.0f}m' if m < 60 else f'{m/60:.1f}h'
        print(f"  Cu{sz:<6} {n:>6}  [{min(dgh_vals):+.2f}, {max(dgh_vals):+.2f}] eV"
              f"  {fmt(t_par):>18}")
    print("  " + "-" * 58)
    print(f"  {'TOTAL':<8} {total_sites:>6}  "
          f"{'':>20}  {fmt(total_time/6):>18}  (6 parallel)")

    print(f"\nNext: python3 04b_gpaw_launch.py")
    return all_prepared


if __name__ == '__main__':
    main()
