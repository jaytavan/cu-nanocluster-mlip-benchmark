"""
06_tensornet_dgh.py

TensorNet ΔGH* — same sites as GPAW benchmark
-----------------------------------------------
Computes ΔGH* with TensorNet (MatPES-PBE-v2025.1-PES via matgl)
on all sites in sites_prepared.json. Same geometry as MACE and GPAW.

Workflow per site:
  1. Reconstruct cluster+H from stored positions
  2. TensorNet single-point → E(cluster+H)
  3. Bare cluster → TensorNet single-point → E(cluster)
  4. E(H2) from TensorNet (computed once, cached)
  5. ΔGH* = E(cluster+H) - E(cluster) - ½·E(H2) + 0.24 eV

Output:
    results/tensornet/results_tensornet.json
    results/tensornet/h2_energy_tensornet.json

Usage:
    python3 06_tensornet_dgh.py
    python3 06_tensornet_dgh.py --test   # 3 sites per size only
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from ase import Atoms

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT       = Path(__file__).parent.parent
RESULTS       = PROJECT / "results"
GPAW_DIR      = RESULTS / "gpaw"
TENSORNET_DIR = RESULTS / "tensornet"
TENSORNET_DIR.mkdir(exist_ok=True)

PREP_JSON = GPAW_DIR / "sites_prepared.json"
OUT_JSON  = TENSORNET_DIR / "results_tensornet.json"
H2_CACHE  = TENSORNET_DIR / "h2_energy_tensornet.json"

MODEL_NAME = "TensorNet-MatPES-PBE-v2025.1-PES"
SIZES      = [10, 20, 30, 40, 50]
ZPE_CORR   = 0.24   # eV


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true',
                   help='Process only 3 sites per size')
    return p.parse_args()


def load_tensornet():
    import torch
    import matgl
    from matgl.ext.ase import PESCalculator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — GPU required.")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"  Loading {MODEL_NAME}...")
    pot = matgl.load_model(MODEL_NAME)
    calc = PESCalculator(pot)
    print(f"  TensorNet ready.")
    return calc


def tensornet_energy(atoms: Atoms, calc) -> float:
    """Run TensorNet single-point, return energy in eV."""
    atoms = atoms.copy()
    atoms.set_pbc(False)
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def compute_h2_tensornet(calc) -> float:
    """Compute E(H2) with TensorNet. Cached."""
    if H2_CACHE.exists():
        with open(H2_CACHE) as f:
            data = json.load(f)
        e_h2 = data['e_h2_eV']
        print(f"  E(H2) TensorNet loaded from cache: {e_h2:.5f} eV")
        return e_h2

    print("  Computing E(H2) with TensorNet...")
    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.center(vacuum=7.5)
    h2.set_pbc(False)

    t0 = time.time()
    e_h2 = tensornet_energy(h2, calc)
    elapsed = time.time() - t0
    print(f"  E(H2) TensorNet = {e_h2:.5f} eV  [{elapsed:.1f}s]")

    with open(H2_CACHE, 'w') as f:
        json.dump({'e_h2_eV': e_h2, 'model': MODEL_NAME}, f, indent=2)
    return e_h2


def rebuild_cluster_h(prepared: dict) -> Atoms:
    cluster_pos = np.array(prepared['cluster_pos'])
    h_pos       = np.array(prepared['h_pos'])
    n_cu        = len(cluster_pos)
    symbols     = ['Cu'] * n_cu + ['H']
    all_pos     = np.vstack([cluster_pos, h_pos])
    atoms       = Atoms(symbols, positions=all_pos)
    atoms.set_pbc(False)
    return atoms


def rebuild_cluster(prepared: dict) -> Atoms:
    cluster_pos = np.array(prepared['cluster_pos'])
    n_cu        = len(cluster_pos)
    atoms       = Atoms(['Cu'] * n_cu, positions=cluster_pos)
    atoms.set_pbc(False)
    return atoms


def main():
    args = parse_args()

    print("=" * 65)
    print("06 — TensorNet ΔGH* (same sites as GPAW benchmark)")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Test mode: {args.test}")
    print("=" * 65)

    if not PREP_JSON.exists():
        raise FileNotFoundError(f"{PREP_JSON} — run 04a_gpaw_prep.py first")
    with open(PREP_JSON) as f:
        all_prepared = json.load(f)

    print(f"\nLoaded {len(all_prepared)} prepared sites from 04a")

    print("\nLoading TensorNet...")
    calc = load_tensornet()

    print("\nStep 1 — E(H2) reference")
    e_h2 = compute_h2_tensornet(calc)

    cluster_energy_cache = {}
    all_results = []
    t_start = time.time()

    for sz in SIZES:
        sites = [p for p in all_prepared if p['size'] == sz]
        if not sites:
            continue

        if args.test:
            sites = sites[:3]

        print(f"\nCu{sz}: {len(sites)} sites")
        print(f"  {'#':>4}  {'ID':<22}  {'Type':<8}  {'ΔGH* TNet':>11}  "
              f"{'ΔGH* MACE':>11}  {'t(s)':>6}")
        print("  " + "-" * 66)

        for i, site in enumerate(sites):
            cid = site['config_id']
            sid = site['site_global_id']
            t_site = time.time()

            try:
                if cid not in cluster_energy_cache:
                    cluster = rebuild_cluster(site)
                    cluster_energy_cache[cid] = tensornet_energy(cluster, calc)

                e_cluster = cluster_energy_cache[cid]
                cluster_h = rebuild_cluster_h(site)
                e_clus_h  = tensornet_energy(cluster_h, calc)

                delta_e = e_clus_h - e_cluster - 0.5 * e_h2
                dgh     = delta_e + ZPE_CORR
                status  = 'ok'
            except Exception as ex:
                print(f"  FAILED: {sid}  {ex}")
                e_cluster, e_clus_h, dgh, status = None, None, None, f'error: {str(ex)[:60]}'

            elapsed = time.time() - t_site

            result = {
                'site_global_id':    sid,
                'config_id':         cid,
                'size':              sz,
                'site_idx':          site['site_idx'],
                'site_type':         site['site_type'],
                'site_atoms':        site['site_atoms'],
                'dgh_mace_eV':       site['dgh_mace_eV'],
                'dgh_tensornet_eV':  round(float(dgh), 5) if dgh is not None else None,
                'delta_dgh_eV':      round(float(dgh - site['dgh_mace_eV']), 5)
                                     if dgh is not None else None,
                'status':            status,
                'time_s':            round(elapsed, 2),
            }
            all_results.append(result)

            if dgh is not None:
                print(f"  {i+1:>4}  {sid:<22}  {site['site_type']:<8}  "
                      f"{dgh:>11.4f}  {site['dgh_mace_eV']:>11.4f}  {elapsed:>6.2f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    good   = [r for r in all_results if r['dgh_tensornet_eV'] is not None]
    failed = [r for r in all_results if r['dgh_tensornet_eV'] is None]

    summary_by_size = {}
    for sz in SIZES:
        group = [r for r in good if r['size'] == sz]
        if not group:
            continue
        dgh_t  = [r['dgh_tensornet_eV'] for r in group]
        dgh_m  = [r['dgh_mace_eV']      for r in group]
        deltas = [r['delta_dgh_eV']     for r in group]
        summary_by_size[sz] = {
            'n_computed':         len(group),
            'dgh_tensornet_min':  round(min(dgh_t), 5),
            'dgh_tensornet_max':  round(max(dgh_t), 5),
            'dgh_mace_min':       round(min(dgh_m), 5),
            'dgh_mace_max':       round(max(dgh_m), 5),
            'mae_vs_mace_eV':     round(float(np.mean(np.abs(deltas))), 5),
            'rmse_vs_mace_eV':    round(float(np.sqrt(np.mean(np.array(deltas)**2))), 5),
        }

    output = {
        'model':           MODEL_NAME,
        'n_sites_total':   len(all_prepared),
        'n_computed':      len(good),
        'n_failed':        len(failed),
        'e_h2_eV':         e_h2,
        'zpe_corr_eV':     ZPE_CORR,
        'total_time_s':    round(time.time() - t_start, 1),
        'summary_by_size': summary_by_size,
        'sites':           all_results,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    wall = output['total_time_s']
    print(f"\n{'='*65}")
    print(f"TensorNet ΔGH* — DONE")
    print(f"  Sites computed: {len(good)} / {len(all_prepared)}")
    print(f"  Failed:         {len(failed)}")
    print(f"  Total time:     {wall:.0f}s")
    print(f"\n  {'Size':<8}  {'Sites':>6}  {'MAE vs MACE':>12}  {'RMSE vs MACE':>13}")
    print("  " + "-" * 44)
    for sz in sorted(summary_by_size):
        s = summary_by_size[sz]
        print(f"  Cu{sz:<6}  {s['n_computed']:>6}  "
              f"{s['mae_vs_mace_eV']:>12.4f}  {s['rmse_vs_mace_eV']:>13.4f}")
    print(f"\n  Results: {OUT_JSON}")
    print(f"  Next: run 07_compare_results.py")


if __name__ == '__main__':
    main()
