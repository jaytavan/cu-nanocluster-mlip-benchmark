"""
03b_atop_sites.py

Atop Site Supplement — MACE ΔGH* for Representative Clusters
--------------------------------------------------------------
Script 03 missed all atop sites because the H-placement overlap guard
(min Cu-H < 1.8 Å) incorrectly rejected legitimate Cu-H bonds (~1.55 Å).
Fixed threshold: 1.4 Å.

This script runs MACE ΔGH* for atop sites ONLY on the 5 representative
clusters (Cu10_00 … Cu50_00) and appends the results to sites_mace.json.

Run AFTER 03_adsorption_sites.py, BEFORE 04a_gpaw_prep.py (or after Cu50
finishes if GPAW is already running).

Output:
  results/sites_mace.json          ← updated in-place (atop sites appended)
  results/sites_mace_atop_only.json ← just the atop results (backup)
"""

import json
import time
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.neighborlist import neighbor_list

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).parent.parent
RESULTS   = PROJECT / "results"
IN_TRAJ   = RESULTS / "clusters_relaxed.traj"
RELAX_LOG = RESULTS / "relaxation_log.json"
OUT_JSON  = RESULTS / "sites_mace.json"
ATOP_JSON = RESULTS / "sites_mace_atop_only.json"

# ── Parameters ─────────────────────────────────────────────────────────────
SIZES          = [10, 20, 30, 40, 50]
REP_SUFFIX     = "_00"
DEVICE         = "cuda"
MACE_MODEL     = "medium"
BOND_CUTOFF    = 3.2
CN_SURFACE_THR = 9
H_DIST         = 1.70   # Å initial placement
H_FMAX         = 0.10   # eV/Å
H_MAX_STEPS    = 150
ZPE_CORR       = 0.24
H_OVERLAP_THR  = 1.4    # Å — fixed from 1.8 to allow Cu-H bonds (~1.55 Å)


def load_mace():
    import torch
    from mace.calculators import mace_mp
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float32")
    print("  MACE ready.\n")
    return calc


def get_surface_atoms(atoms):
    i_idx, _ = neighbor_list('ij', atoms, BOND_CUTOFF)
    cn = np.bincount(i_idx, minlength=len(atoms))
    return [int(i) for i in range(len(atoms)) if cn[i] <= CN_SURFACE_THR], cn


def place_h_atop(atoms, site_pos, h_dist=H_DIST):
    center    = atoms.get_positions().mean(axis=0)
    direction = site_pos - center
    norm      = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0., 0., 1.])
    else:
        direction /= norm

    h_pos  = site_pos + h_dist * direction
    site_r = np.linalg.norm(site_pos - center)
    h_r    = np.linalg.norm(h_pos    - center)

    if h_r < site_r - 0.2:
        return None   # concave

    dists = np.linalg.norm(atoms.get_positions() - h_pos, axis=1)
    if np.min(dists) < H_OVERLAP_THR:
        return None   # too close to Cu

    cluster_h = atoms.copy()
    cluster_h += Atoms('H', positions=[h_pos])
    cluster_h.set_pbc(False)
    return cluster_h


def relax_h(cluster_h, calc):
    n = len(cluster_h) - 1
    cluster_h = cluster_h.copy()
    cluster_h.set_constraint(FixAtoms(indices=list(range(n))))
    cluster_h.calc = calc
    opt = LBFGS(cluster_h, logfile=None)
    conv = opt.run(fmax=H_FMAX, steps=H_MAX_STEPS)
    return cluster_h, float(cluster_h.get_potential_energy()), bool(conv)


def main():
    print("=" * 65)
    print("03b — Atop Sites (MACE ΔGH*)")
    print(f"  H overlap threshold: {H_OVERLAP_THR} Å (was 1.8)")
    print("=" * 65)

    clusters_all = read(str(IN_TRAJ), index=':')
    cluster_map  = {a.info.get('config_id'): a for a in clusters_all}

    with open(RELAX_LOG) as f:
        relax_logs = json.load(f)
    e_cluster_map = {l['config_id']: l['e_after_eV']
                     for l in relax_logs if 'e_after_eV' in l}

    with open(OUT_JSON) as f:
        existing = json.load(f)

    print("Loading MACE...")
    mace_calc = load_mace()

    # Compute E(H2) with MACE
    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.center(vacuum=7.5)
    h2.set_pbc(False)
    h2.calc = mace_calc
    e_h2 = float(h2.get_potential_energy())
    print(f"E(H2) MACE = {e_h2:.5f} eV\n")

    all_atop = {}   # cid → list of site dicts

    for sz in SIZES:
        cid = f"Cu{sz}{REP_SUFFIX}"
        cluster = cluster_map.get(cid)
        if cluster is None:
            print(f"  WARNING: {cid} not found — skipping")
            continue

        e_cluster = e_cluster_map.get(cid)
        surf_idx, cn = get_surface_atoms(cluster)
        pos = cluster.get_positions()

        print(f"\nCu{sz}_00: {len(surf_idx)} surface atoms → atop sites")

        sites_out = []
        n_skip = 0
        t0 = time.time()

        for i_surf, atom_idx in enumerate(surf_idx):
            site_pos  = pos[atom_idx].copy()
            cluster_h = place_h_atop(cluster, site_pos)

            if cluster_h is None:
                n_skip += 1
                continue

            cluster_h, e_clus_h, conv = relax_h(cluster_h, mace_calc)

            # H position after relaxation
            h_pos_relaxed = cluster_h.get_positions()[-1]

            # Sanity check: H should not have migrated far from original atom
            h_dist_to_atom = np.linalg.norm(h_pos_relaxed - site_pos)

            dgh = (e_clus_h - e_cluster - 0.5 * e_h2) + ZPE_CORR

            sites_out.append({
                'type':       'atop',
                'atoms':      [atom_idx],
                'dgh_eV':     round(float(dgh), 5),
                'e_clus_h':   round(float(e_clus_h), 5),
                'e_h2_mace':  e_h2,
                'converged':  conv,
                'h_drift_A':  round(float(h_dist_to_atom), 3),
            })

        elapsed = time.time() - t0
        dgh_vals = [s['dgh_eV'] for s in sites_out]
        n_pass   = sum(1 for v in dgh_vals if abs(v) < 0.5)

        print(f"  {len(sites_out)} atop sites computed  ({n_skip} skipped)  {elapsed:.0f}s")
        if dgh_vals:
            print(f"  ΔGH* range: [{min(dgh_vals):.3f}, {max(dgh_vals):.3f}] eV")
            print(f"  Sites with |ΔGH*| < 0.5 eV: {n_pass} / {len(sites_out)}")

        all_atop[cid] = sites_out

    # Release GPU
    del mace_calc
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    # Save atop-only backup
    with open(ATOP_JSON, 'w') as f:
        json.dump(all_atop, f, indent=2)
    print(f"\nAtop-only results saved: {ATOP_JSON}")

    # Append to existing sites_mace.json
    for cid, atop_sites in all_atop.items():
        if cid in existing:
            # Check for duplicates (in case of re-run)
            existing_types = {s['type'] for s in existing[cid]['sites']}
            if 'atop' in existing_types:
                print(f"  {cid}: atop sites already in sites_mace.json — skipping append")
                continue
            existing[cid]['sites'].extend(atop_sites)
            existing[cid]['n_sites'] = len(existing[cid]['sites'])
            # Update summary stats
            dgh_all = [s['dgh_eV'] for s in existing[cid]['sites']]
            existing[cid]['dgh_min_eV']  = round(min(dgh_all), 5)
            existing[cid]['dgh_max_eV']  = round(max(dgh_all), 5)
            existing[cid]['dgh_mean_eV'] = round(float(np.mean(dgh_all)), 5)

    with open(OUT_JSON, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"sites_mace.json updated with atop sites.\n")

    # Summary
    print(f"\n{'='*65}")
    print("Atop site summary:")
    print(f"  {'Size':<8} {'Total':>6}  {'|ΔGH*|<0.5':>11}  {'ΔGH* range':>20}")
    print("  " + "-" * 50)
    total_atop = 0
    total_pass = 0
    for sz in SIZES:
        cid = f"Cu{sz}{REP_SUFFIX}"
        sites = all_atop.get(cid, [])
        if not sites:
            continue
        dgh_vals = [s['dgh_eV'] for s in sites]
        n_pass   = sum(1 for v in dgh_vals if abs(v) < 0.5)
        total_atop += len(sites)
        total_pass += n_pass
        print(f"  Cu{sz:<6} {len(sites):>6}  {n_pass:>11}  "
              f"[{min(dgh_vals):+.2f}, {max(dgh_vals):+.2f}] eV")
    print("  " + "-" * 50)
    print(f"  {'TOTAL':<8} {total_atop:>6}  {total_pass:>11}")
    print(f"\nNext: run 04a_gpaw_prep.py after Cu50 GPAW finishes to include atop sites.")


if __name__ == '__main__':
    main()
