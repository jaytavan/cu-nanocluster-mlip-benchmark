"""
04c_append_cu10_atop.py

Appends Cu10 atop sites to sites_prepared.json without touching existing data.

Cu10 currently has 6 hollow sites (|ΔGH*| < 0.5 eV, site000-site005).
This script adds 9 atop sites (|ΔGH*| < 0.75 eV, site006-site014).

These were excluded from the original prep because the global filter was 0.5 eV.
Cu10 atop sites cluster at +0.52, +0.60, and +0.66 eV — all physically valid,
useful for testing model accuracy on higher-ΔGH* atop environments.

Run once, then:
    python3 scripts/04b_gpaw_launch.py --sizes 10 --ncores 12 --sequential
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).parent.parent
RESULTS_DIR   = PROJECT_DIR / "results"
GPAW_DIR      = RESULTS_DIR / "gpaw"
SITES_JSON    = RESULTS_DIR / "sites_mace.json"
CLUSTERS_TRAJ = RESULTS_DIR / "clusters_relaxed.traj"
RELAX_LOG     = RESULTS_DIR / "relaxation_log.json"
PREP_JSON     = GPAW_DIR / "sites_prepared.json"

DGH_FILTER = 0.75   # eV — wider than default 0.5 to capture Cu10 atop groups
ZPE_CORR   = 0.24   # eV — standard CHE correction
CID        = "Cu10_00"
SZ         = 10

# ── MACE loader ───────────────────────────────────────────────────────────────
def load_mace():
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
    print("  MACE-MP-0 medium loaded on CUDA.")
    return calc


def relax_h_mace(cluster, site_centroid, calc):
    """Place H ~1.6 Å above site centroid toward vacuum, relax with MACE."""
    from ase import Atoms
    from ase.optimize import BFGS
    import warnings

    cu_pos = cluster.get_positions()
    cluster_centre = cu_pos.mean(axis=0)
    direction = site_centroid - cluster_centre
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction /= norm

    h_pos = site_centroid + 1.6 * direction

    symbols = list(cluster.get_chemical_symbols()) + ["H"]
    positions = np.vstack([cu_pos, h_pos])
    cluster_h = Atoms(symbols=symbols, positions=positions)
    cluster_h.center(vacuum=8.0)
    cluster_h.set_pbc(True)
    cluster_h.calc = calc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = BFGS(cluster_h, logfile=None)
        opt.run(fmax=0.05, steps=200)

    return cluster_h


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("04c — Append Cu10 atop sites to sites_prepared.json")
    print(f"  Filter: |ΔGH*_MACE| < {DGH_FILTER} eV  (atop sites only)")
    print("=" * 65)

    # Load existing prep — must not touch existing entries
    existing = json.load(open(PREP_JSON))
    existing_ids = {s["site_global_id"] for s in existing}
    existing_cu10 = [s for s in existing if s["config_id"] == CID]
    next_idx = max((s["site_idx"] for s in existing_cu10), default=-1) + 1
    print(f"\nExisting Cu10 sites: {len(existing_cu10)} (indices 0–{next_idx-1})")
    print(f"New sites will start at site_idx={next_idx}")

    # Load cluster geometry
    from ase.io import read
    clusters_all = read(str(CLUSTERS_TRAJ), index=":")
    cluster_map = {a.info.get("config_id"): a for a in clusters_all}
    cluster = cluster_map.get(CID)
    if cluster is None:
        print(f"ERROR: {CID} not found in clusters_relaxed.traj")
        sys.exit(1)

    relax_logs = json.load(open(RELAX_LOG))
    e_cluster_map = {l["config_id"]: l["e_after_eV"]
                     for l in relax_logs if "e_after_eV" in l}
    e_cluster = e_cluster_map.get(CID)

    e_h2_cache = GPAW_DIR / "h2_energy_gpaw.json"
    e_h2_gpaw = json.load(open(e_h2_cache))["e_h2_eV"]

    # Load Cu10 sites from sites_mace.json
    sites_data = json.load(open(SITES_JSON))
    cu10_sites = sites_data[CID]["sites"]
    atop_candidates = [s for s in cu10_sites
                       if s.get("type") == "atop" and abs(s.get("dgh_eV", 99)) < DGH_FILTER]
    print(f"Cu10 atop candidates (|ΔGH*|<{DGH_FILTER}): {len(atop_candidates)}")
    for s in sorted(atop_candidates, key=lambda x: x["dgh_eV"]):
        print(f"  {s['dgh_eV']:+.4f} eV")

    # MACE H relaxation
    print("\nLoading MACE...")
    calc = load_mace()

    print(f"\nRelaxing H for {len(atop_candidates)} atop sites...")
    new_entries = []
    for i, site in enumerate(sorted(atop_candidates, key=lambda x: x["dgh_eV"])):
        idx = next_idx + i
        gid = f"{CID}_site{idx:03d}"

        # Skip if already prepared (safety check)
        if gid in existing_ids:
            print(f"  [{i+1}/{len(atop_candidates)}] {gid} already exists — skip")
            continue

        cu_pos = cluster.get_positions()
        atom_indices = site["atoms"]
        site_centroid = cu_pos[atom_indices].mean(axis=0)

        t0 = time.time()
        cluster_h = relax_h_mace(cluster, site_centroid, calc)
        elapsed = time.time() - t0

        h_pos = cluster_h.get_positions()[-1]

        entry = {
            "site_global_id": gid,
            "config_id":      CID,
            "size":           SZ,
            "site_idx":       idx,
            "site_type":      "atop",
            "site_atoms":     site["atoms"],
            "dgh_mace_eV":    site["dgh_eV"],
            "e_cluster_mace": e_cluster,
            "e_h2_gpaw":      e_h2_gpaw,
            "h_pos":          h_pos.tolist(),
            "cluster_pos":    cluster.get_positions().tolist(),
            "cell":           cluster_h.get_cell().tolist(),
            "zpe_corr":       ZPE_CORR,
            "mace_h_relax_s": round(elapsed, 2),
        }
        new_entries.append(entry)
        print(f"  [{i+1}/{len(atop_candidates)}] {gid}  ΔGH*={site['dgh_eV']:+.4f} eV  "
              f"H-relax: {elapsed:.1f}s")

    del calc
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    # Append and save
    combined = existing + new_entries
    with open(PREP_JSON, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nAppended {len(new_entries)} Cu10 atop sites.")
    print(f"sites_prepared.json now has {len(combined)} entries total.")
    print(f"\nNext: python3 scripts/04b_gpaw_launch.py --sizes 10 --ncores 12 --sequential")


if __name__ == "__main__":
    main()
