"""
04d_phase_c_prep.py

Phase C — GPAW Preparation for Additional Cluster Geometries
-------------------------------------------------------------
Extends sites_prepared.json with MACE H-relaxed sites for Cu{N}_01
and Cu{N}_02 (bridge + hollow only) across all five cluster sizes.

Phase A used only Cu{N}_00 (one representative per size, 145 GPAW sites).
Phase C adds 2 more geometries per size to test whether the benchmark
conclusions hold across different local minima, increasing reviewer confidence.

Steps (mirrors 04a_gpaw_prep.py, extended to non-representative clusters):
  1. Load MACE-relaxed cluster geometries for _01 and _02 from clusters_relaxed.traj
  2. Load adsorption sites from sites_mace.json
  3. Filter: bridge + hollow only, |ΔGH*_MACE| < 0.5 eV, converged=True
  4. Re-relax H position with MACE (frozen cluster) for each passing site
  5. Append new entries to existing sites_prepared.json (preserves _00 data)

Run before 04b_gpaw_launch.py --cluster-ids 01 02:
    python3 scripts/04d_phase_c_prep.py
    python3 scripts/04b_gpaw_launch.py --cluster-ids 01 02 --ncores 12 --sequential

Prerequisites:
    results/clusters_relaxed.traj       (all 30 MACE-relaxed clusters)
    results/sites_mace.json             (all 30×N bridge/hollow/atop sites)
    results/relaxation_log.json         (MACE cluster energies)
    results/gpaw/h2_energy_gpaw.json    (GPAW H2 reference, from 04a)
    results/gpaw/sites_prepared.json    (existing Phase A entries — will be extended)
"""

import json
import time
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT       = Path(__file__).parent.parent
RESULTS       = PROJECT / "results"
GPAW_DIR      = RESULTS / "gpaw"

CLUSTERS_TRAJ = RESULTS / "clusters_relaxed.traj"
SITES_JSON    = RESULTS / "sites_mace.json"
RELAX_LOG     = RESULTS / "relaxation_log.json"
PREP_JSON     = GPAW_DIR / "sites_prepared.json"
H2_CACHE      = GPAW_DIR / "h2_energy_gpaw.json"

# ── Parameters ─────────────────────────────────────────────────────────────────
SIZES      = [10, 20, 30, 40, 50]
PHASE_C_IDS = ["01", "02"]             # cluster IDs to add
SITE_TYPES  = {"bridge", "hollow"}     # atop excluded for Phase C
DGH_FILTER  = 0.5                      # eV — same as Phase A
DEVICE      = "cuda"
MACE_MODEL  = "medium"
H_FMAX      = 0.10                     # eV/Å
H_MAX_STEPS = 150
ZPE_CORR    = 0.24                     # eV


def load_mace():
    import torch
    from mace.calculators import mace_mp
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float32")
    print("  MACE ready.")
    return calc


def relax_h_mace(cluster: Atoms, site_centroid: np.ndarray, mace_calc) -> Atoms:
    """Place H 1.7 Å above site centroid along radial direction, then relax."""
    center    = cluster.get_positions().mean(axis=0)
    direction = site_centroid - center
    norm      = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    h_pos = site_centroid + 1.7 * direction

    cluster_h = cluster.copy()
    cluster_h += Atoms("H", positions=[h_pos])
    cluster_h.set_pbc(False)
    cluster_h.set_constraint(FixAtoms(indices=list(range(len(cluster)))))
    cluster_h.calc = mace_calc

    opt = LBFGS(cluster_h, logfile=None)
    opt.run(fmax=H_FMAX, steps=H_MAX_STEPS)
    cluster_h.set_constraint()
    return cluster_h


def main():
    print("=" * 65)
    print("04d — Phase C GPAW Site Preparation")
    print(f"  Cluster IDs: {PHASE_C_IDS}  (appending to Phase A _00 data)")
    print(f"  Site types:  bridge + hollow  (no atop)")
    print(f"  ΔGH* filter: |ΔGH*_MACE| < {DGH_FILTER} eV")
    print("=" * 65)

    # ── Sanity checks ────────────────────────────────────────────────────────
    for f in [CLUSTERS_TRAJ, SITES_JSON, RELAX_LOG, H2_CACHE, PREP_JSON]:
        if not f.exists():
            raise FileNotFoundError(f"Required file missing: {f}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading input data...")
    clusters_all = read(str(CLUSTERS_TRAJ), index=":")
    cluster_map  = {a.info.get("config_id"): a for a in clusters_all}

    sites_data = json.load(open(SITES_JSON))      # {cid: {sites: [...]}}
    relax_logs = json.load(open(RELAX_LOG))
    e_cluster_map = {l["config_id"]: l["e_after_eV"]
                     for l in relax_logs if "e_after_eV" in l}

    e_h2_gpaw = json.load(open(H2_CACHE))["e_h2_eV"]
    print(f"  E(H2) GPAW = {e_h2_gpaw:.5f} eV  (from cache)")

    existing = json.load(open(PREP_JSON))
    existing_cids = set(s["config_id"] for s in existing)
    print(f"  Existing sites_prepared.json: {len(existing)} entries "
          f"({sorted(existing_cids)})")

    # ── Check for already-prepared Phase C entries ───────────────────────────
    phase_c_cids = [f"Cu{sz}_{cid}" for sz in SIZES for cid in PHASE_C_IDS]
    already_done = [cid for cid in phase_c_cids if cid in existing_cids]
    to_run       = [cid for cid in phase_c_cids if cid not in existing_cids]

    if already_done:
        print(f"  Already prepared: {already_done} — skipping these")
    if not to_run:
        print("  All Phase C clusters already in sites_prepared.json. Nothing to do.")
        return
    print(f"  Will prepare: {to_run}")

    # ── Site summary ─────────────────────────────────────────────────────────
    print("\n[2] Site count preview:")
    total_sites = 0
    for cid in to_run:
        if cid not in sites_data:
            print(f"  WARNING: {cid} not in sites_mace.json — skip")
            continue
        cluster_sites = sites_data[cid]["sites"]
        passable = [s for s in cluster_sites
                    if s["type"] in SITE_TYPES
                    and s.get("converged", True)
                    and abs(s["dgh_eV"]) < DGH_FILTER]
        total_sites += len(passable)
        print(f"  {cid}: {len(cluster_sites)} total → {len(passable)} pass filter")
    print(f"  Total to prepare: {total_sites} sites")

    # ── Load MACE ────────────────────────────────────────────────────────────
    print("\n[3] Loading MACE...")
    mace_calc = load_mace()

    # ── H relaxation loop ────────────────────────────────────────────────────
    print("\n[4] Running MACE H relaxation...")
    new_prepared = []
    t_total = time.time()

    for cid in to_run:
        if cid not in sites_data:
            continue

        sz  = int(cid.split("_")[0][2:])
        cluster = cluster_map.get(cid)
        if cluster is None:
            print(f"  WARNING: {cid} not in clusters_relaxed.traj — skip")
            continue

        e_cluster_mace = e_cluster_map.get(cid)
        cluster_sites  = sites_data[cid]["sites"]

        filtered = [s for s in cluster_sites
                    if s["type"] in SITE_TYPES
                    and s.get("converged", True)
                    and abs(s["dgh_eV"]) < DGH_FILTER]

        print(f"\n  {cid}: {len(filtered)} sites to prepare")
        t0_cid = time.time()

        for i, site in enumerate(filtered):
            cu_pos        = cluster.get_positions()
            atom_indices  = site["atoms"]
            site_centroid = cu_pos[atom_indices].mean(axis=0)

            t_site = time.time()
            cluster_h = relax_h_mace(cluster, site_centroid, mace_calc)
            elapsed   = time.time() - t_site

            h_pos = cluster_h.get_positions()[-1]

            entry = {
                "site_global_id": f"{cid}_site{i:03d}",
                "config_id":      cid,
                "size":           sz,
                "site_idx":       i,
                "site_type":      site["type"],
                "site_atoms":     site["atoms"],
                "dgh_mace_eV":    float(site["dgh_eV"]),
                "e_cluster_mace": e_cluster_mace,
                "e_h2_gpaw":      e_h2_gpaw,
                "h_pos":          h_pos.tolist(),
                "cluster_pos":    cluster.get_positions().tolist(),
                "cell":           cluster.get_cell().tolist(),
                "zpe_corr":       ZPE_CORR,
                "mace_h_relax_s": round(elapsed, 2),
            }
            new_prepared.append(entry)

            if (i + 1) % 10 == 0 or i == len(filtered) - 1:
                print(f"    [{i+1}/{len(filtered)}] {site['type']:<7} "
                      f"ΔGH*={site['dgh_eV']:+.3f} eV  H-relax: {elapsed:.1f}s")

        elapsed_cid = time.time() - t0_cid
        print(f"  {cid} done: {len(filtered)} sites in {elapsed_cid:.1f}s")

    # ── Release GPU ──────────────────────────────────────────────────────────
    del mace_calc
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    print("\n  MACE released.")

    # ── Append to sites_prepared.json ────────────────────────────────────────
    print(f"\n[5] Appending {len(new_prepared)} new entries to sites_prepared.json...")
    updated = existing + new_prepared
    json.dump(updated, open(PREP_JSON, "w"), indent=2)

    total_elapsed = time.time() - t_total
    print(f"  Done. sites_prepared.json now has {len(updated)} entries.")
    print(f"  Total MACE prep time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PHASE C PREP SUMMARY")
    print(f"{'='*65}")
    for sz in SIZES:
        for cid_suf in PHASE_C_IDS:
            cid   = f"Cu{sz}_{cid_suf}"
            group = [s for s in new_prepared if s["config_id"] == cid]
            if not group:
                continue
            dgh_vals = [s["dgh_mace_eV"] for s in group]
            print(f"  {cid}: {len(group)} sites  "
                  f"ΔGH* [{min(dgh_vals):+.3f}, {max(dgh_vals):+.3f}] eV")

    print(f"\nTotal new sites queued for GPAW: {len(new_prepared)}")
    print(f"\nNext:")
    print(f"  python3 scripts/04b_gpaw_launch.py "
          f"--cluster-ids 01 02 --ncores 12 --sequential")


if __name__ == "__main__":
    main()
