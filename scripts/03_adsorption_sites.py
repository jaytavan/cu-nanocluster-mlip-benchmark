"""
03_adsorption_sites.py

Phase 1 — H* Site Identification & MACE ΔGH* Calculation
----------------------------------------------------------
For each MACE-relaxed pure Cu nanocluster:
  1. Identify surface atoms (coordination number < CN_SURFACE_THR)
  2. Find unique adsorption sites: atop, bridge, 3-fold hollow
  3. Place H at each site, relax H-only (cluster frozen), compute E(cluster+H)
  4. ΔG_H = [E(cluster+H) - E(cluster) - ½·E(H2)] + 0.24 eV (standard CHE)

Design note — WHY this script uses general surface-atom detection:
  The prior project used icosahedral V/E/C shell classification, which is
  specific to Cu55. For this benchmark we have multiple cluster sizes and
  geometries that may not be icosahedral after MACE relaxation. A general
  coordination-number approach works for any shape or size.

Surface atom detection:
  - Coordination number: count neighbors within BOND_CUTOFF = 3.2 Å
  - Cu bulk FCC: CN = 12. Surface atoms have CN ≤ 9.
  - This threshold separates surface from interior atoms for sizes 10–50.

Site types:
  - Atop:   directly above one surface atom (most common)
  - Bridge: midpoint of two bonded surface atoms (within SURF_BOND_CUTOFF)
  - Hollow: centroid of three mutually bonded surface atoms

H placement: 1.7 Å outward from site centroid (along radial direction)
Cluster atoms: frozen during H relaxation (MACE calculator on full system)

Output:
  results/sites_mace.json          — full per-site ΔGH* results
  results/sites_summary_mace.json  — per-cluster summary (size, n_sites, min/mean/max ΔGH*)
  results/clusters_with_h.traj     — trajectory of all cluster+H relaxed structures
"""

import json
import time
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.neighborlist import neighbor_list

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT    = Path(__file__).parent.parent
RESULTS    = PROJECT / "results"
IN_TRAJ    = RESULTS / "clusters_relaxed.traj"
RELAX_LOG  = RESULTS / "relaxation_log.json"
OUT_JSON   = RESULTS / "sites_mace.json"
SUM_JSON   = RESULTS / "sites_summary_mace.json"
OUT_H_TRAJ = RESULTS / "clusters_with_h.traj"

# ── Parameters ─────────────────────────────────────────────────────────────
DEVICE         = "cuda"
MACE_MODEL     = "medium"
BOND_CUTOFF    = 3.2   # Å — neighbor search cutoff for CN calculation
CN_SURFACE_THR = 9     # atoms with CN ≤ this are surface atoms
SURF_BOND_CUT  = 3.2   # Å — max distance to consider two surface atoms bonded
H_DIST         = 1.70  # Å — initial H distance from site centroid (outward)
H_FMAX         = 0.10  # eV/Å — looser convergence for H-only relaxation
H_MAX_STEPS    = 150
ZPE_CORR       = 0.24  # eV — ΔG_H = ΔE_H + 0.24 eV (standard CHE at 298K, 1 bar)


# ── Surface atom detection ───────────────────────────────────────────────────
def get_surface_atoms(atoms: Atoms, bond_cutoff: float = BOND_CUTOFF,
                      cn_thr: int = CN_SURFACE_THR) -> list:
    """
    Return indices of surface atoms: those with coordination number ≤ cn_thr.

    Uses ASE neighbor_list for efficiency. For pure Cu clusters:
      bulk FCC: CN = 12
      surface:  CN ≤ 9  (empirically good for Cu10–Cu50)
    """
    i_idx, _ = neighbor_list('ij', atoms, bond_cutoff)
    cn = np.bincount(i_idx, minlength=len(atoms))
    surface = [int(i) for i in range(len(atoms)) if cn[i] <= cn_thr]
    return surface, cn


# ── Site finders ─────────────────────────────────────────────────────────────
def find_atop_sites(atoms: Atoms, surface_idx: list) -> list:
    """
    Atop sites: directly above each surface atom.
    Returns list of ('atop', atom_idx, site_position) tuples.
    """
    pos = atoms.get_positions()
    return [('atop', i, pos[i].copy()) for i in surface_idx]


def find_bridge_sites(atoms: Atoms, surface_idx: list,
                      bond_cut: float = SURF_BOND_CUT) -> list:
    """
    Bridge sites: midpoint of each bonded pair of surface atoms.
    Returns list of ('bridge', i, j, midpoint) tuples.
    """
    pos = atoms.get_positions()
    surf_set = set(surface_idx)
    sites = []
    for a, i in enumerate(surface_idx):
        for b, j in enumerate(surface_idx):
            if j <= i:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= bond_cut:
                midpoint = 0.5 * (pos[i] + pos[j])
                sites.append(('bridge', i, j, midpoint))
    return sites


def find_hollow_sites(atoms: Atoms, surface_idx: list,
                      bond_cut: float = SURF_BOND_CUT) -> list:
    """
    3-fold hollow sites: centroid of three mutually bonded surface atoms.
    Returns list of ('hollow', i, j, k, centroid) tuples.
    """
    pos = atoms.get_positions()
    n_surf = len(surface_idx)
    sites = []
    for a in range(n_surf):
        i = surface_idx[a]
        for b in range(a + 1, n_surf):
            j = surface_idx[b]
            if np.linalg.norm(pos[i] - pos[j]) > bond_cut:
                continue
            for c in range(b + 1, n_surf):
                k = surface_idx[c]
                if (np.linalg.norm(pos[i] - pos[k]) <= bond_cut and
                        np.linalg.norm(pos[j] - pos[k]) <= bond_cut):
                    centroid = (pos[i] + pos[j] + pos[k]) / 3.0
                    sites.append(('hollow', i, j, k, centroid))
    return sites


# ── H placement & relaxation ─────────────────────────────────────────────────
def outward_unit_vector(cluster_center: np.ndarray, site_pos: np.ndarray) -> np.ndarray:
    """Unit vector from cluster center through adsorption site (places H outward)."""
    vec = site_pos - cluster_center
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def place_h_atom(atoms: Atoms, site_pos: np.ndarray, h_dist: float = H_DIST):
    """
    Create cluster+H system with H placed h_dist Å outward from site_pos.
    H is appended as the last atom.

    Returns None (skip site) if:
      1. The site is concave/interior: placing H outward would move it CLOSER to
         the cluster center than site_pos itself. This catches hollow sites on
         the inner concave face of small clusters (Cu20/Cu30 where all atoms
         are classified as surface).
      2. H would overlap a Cu atom (distance < 1.8 Å). Prevents unphysical
         starting geometries that yield enormous spurious energies.
    """
    center    = atoms.get_positions().mean(axis=0)
    direction = outward_unit_vector(center, site_pos)
    h_pos     = site_pos + h_dist * direction

    # Guard 1: H must be farther from cluster center than site_pos
    site_r = np.linalg.norm(site_pos - center)
    h_r    = np.linalg.norm(h_pos    - center)
    if h_r < site_r - 0.2:          # 0.2 Å tolerance
        return None                  # concave site — skip

    # Guard 2: H must not overlap any Cu atom
    # Threshold 1.4 Å (not 1.8): allows legitimate Cu-H bonds (~1.55 Å for atop)
    # while still rejecting H deeply embedded inside the cluster.
    dists_to_cu = np.linalg.norm(atoms.get_positions() - h_pos, axis=1)
    if np.min(dists_to_cu) < 1.4:
        return None                  # H inside cluster — skip

    cluster_h = atoms.copy()
    cluster_h += Atoms('H', positions=[h_pos])
    cluster_h.set_pbc(False)
    return cluster_h


def relax_h_frozen_cluster(cluster_h: Atoms, calc, fmax: float = H_FMAX,
                            max_steps: int = H_MAX_STEPS) -> tuple:
    """
    Fix all cluster atoms, relax only the H atom (last atom in cluster_h).
    Returns (relaxed_atoms, total_energy, converged).
    """
    n_cluster = len(cluster_h) - 1
    cluster_h = cluster_h.copy()
    cluster_h.set_constraint(FixAtoms(indices=list(range(n_cluster))))
    cluster_h.calc = calc

    opt       = LBFGS(cluster_h, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    energy    = cluster_h.get_potential_energy()

    return cluster_h, float(energy), bool(converged)


# ── ΔGH* ─────────────────────────────────────────────────────────────────────
def compute_dgh(e_cluster_h: float, e_cluster: float, e_h2: float,
                zpe_corr: float = ZPE_CORR) -> float:
    """
    ΔG_H = ΔE_H + ΔEZPE - TΔS
         = [E(cluster+H) - E(cluster) - ½·E(H2)] + 0.24 eV

    The 0.24 eV correction (Norskov CHE) accounts for ZPE difference between
    H* and ½H2 (gas), plus entropic contribution at standard conditions.
    """
    delta_e = e_cluster_h - e_cluster - 0.5 * e_h2
    return delta_e + zpe_corr


# ── H2 reference ─────────────────────────────────────────────────────────────
def compute_h2_reference(calc) -> float:
    """
    Relax H2 molecule and return its energy.
    This is the MACE H2 reference — must match the calculator used for clusters.
    """
    h2 = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    h2.center(vacuum=7.5)
    h2.set_pbc(False)
    h2.calc = calc

    opt = LBFGS(h2, logfile=None)
    opt.run(fmax=0.01, steps=100)

    e_h2      = h2.get_potential_energy()
    bond_len  = h2.get_distance(0, 1)
    print(f"  E(H2) MACE = {e_h2:.5f} eV  (bond: {bond_len:.3f} Å, expect ~0.74 Å)")
    return float(e_h2)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Phase 1 — H* Adsorption Sites & MACE ΔGH*")
    print(f"  Surface atom CN threshold: ≤ {CN_SURFACE_THR}")
    print(f"  Site types: atop, bridge, 3-fold hollow")
    print(f"  H placement distance: {H_DIST} Å (outward from centroid)")
    print(f"  H relaxation fmax:    {H_FMAX} eV/Å  (cluster frozen)")
    print(f"  ZPE correction:       +{ZPE_CORR} eV  (standard CHE, 298K, 1 bar)")
    print("=" * 70)

    if not IN_TRAJ.exists():
        raise FileNotFoundError(f"{IN_TRAJ} — run 02_mace_relax.py first")

    structures = read(str(IN_TRAJ), index=':')
    with open(RELAX_LOG) as f:
        relax_logs = json.load(f)

    # Build lookup: config_id → relaxed energy (avoid MACE recalculation for clusters)
    e_cluster_map = {
        l['config_id']: l['e_after_eV']
        for l in relax_logs if 'e_after_eV' in l
    }

    print(f"\nLoaded {len(structures)} relaxed clusters")

    # ── Load MACE ──────────────────────────────────────────────────────────
    import torch
    from mace.calculators import mace_mp

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — abort.")
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Loading MACE-MP-0 ({MACE_MODEL})...")
    calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float32")
    print("  MACE ready.\n")

    # ── H2 reference energy ────────────────────────────────────────────────
    print("Computing MACE E(H2) reference...")
    e_h2 = compute_h2_reference(calc)
    print()

    # ── Main loop ──────────────────────────────────────────────────────────
    all_results     = {}
    all_h_structs   = []
    summary_rows    = []

    print(f"  {'ID':<20} {'N':>3} {'Surf':>5} {'Valid':>6}{'Skip':<12} "
          f"{'ΔGH min':>9} {'ΔGH max':>9} {'ΔGH avg':>9} {'t(s)':>6}")
    print("  " + "-" * 80)

    for atoms in structures:
        config_id = atoms.info.get('config_id', 'unknown')
        n_atoms   = len(atoms)
        size      = atoms.info.get('size', n_atoms)

        # Cluster energy from relaxation log (saved by 02_mace_relax.py)
        e_cluster = e_cluster_map.get(config_id)
        if e_cluster is None:
            # Fallback: recompute (slow — only if log entry missing)
            print(f"  WARNING: no log entry for {config_id}, recomputing energy...")
            atoms.calc = calc
            e_cluster = float(atoms.get_potential_energy())

        # ── Surface atom detection ─────────────────────────────────────────
        surface_idx, cn = get_surface_atoms(atoms)
        n_surf = len(surface_idx)

        # ── Site finding ───────────────────────────────────────────────────
        atop_sites   = find_atop_sites(atoms, surface_idx)
        bridge_sites = find_bridge_sites(atoms, surface_idx)
        hollow_sites = find_hollow_sites(atoms, surface_idx)

        all_sites = (
            [(s[0], s[1], s[2]) for s in atop_sites] +         # (type, i, pos)
            [(s[0], s[1], s[3]) for s in bridge_sites] +       # (type, i, pos)
            [(s[0], s[1], s[4]) for s in hollow_sites]         # (type, i, pos)
        )
        # Richer tuples for JSON output
        sites_rich = (
            [{'type': 'atop',   'atoms': [s[1]],        'pos': s[2].tolist()} for s in atop_sites] +
            [{'type': 'bridge', 'atoms': [s[1], s[2]],  'pos': s[3].tolist()} for s in bridge_sites] +
            [{'type': 'hollow', 'atoms': [s[1], s[2], s[3]], 'pos': s[4].tolist()} for s in hollow_sites]
        )

        n_sites   = len(sites_rich)
        t0        = time.time()
        site_results = []

        n_skipped = 0
        for site_meta, site_info in zip(sites_rich, all_sites):
            stype, _, site_pos = site_info
            site_pos = np.array(site_info[2])

            cluster_h = place_h_atom(atoms, site_pos)
            if cluster_h is None:
                n_skipped += 1
                continue  # concave/interior site — skip

            cluster_h_relax, e_clus_h, conv = relax_h_frozen_cluster(cluster_h, calc)
            dgh = compute_dgh(e_clus_h, e_cluster, e_h2)

            # Final H position
            h_pos_final = cluster_h_relax.get_positions()[-1]
            h_disp = float(np.linalg.norm(h_pos_final - (site_pos + H_DIST *
                           outward_unit_vector(atoms.get_positions().mean(axis=0), site_pos))))

            site_result = {
                'type':       site_meta['type'],
                'atoms':      site_meta['atoms'],
                'dgh_eV':     round(float(dgh), 5),
                'e_clus_h':   round(float(e_clus_h), 5),
                'converged':  conv,
                'h_displ_A':  round(h_disp, 3),
            }
            site_results.append(site_result)
            all_h_structs.append(cluster_h_relax)

        elapsed = time.time() - t0
        n_valid = len(site_results)

        dgh_vals = [s['dgh_eV'] for s in site_results]

        cluster_result = {
            'config_id':    config_id,
            'size':         size,
            'n_atoms':      n_atoms,
            'e_cluster_eV': round(float(e_cluster), 5),
            'e_h2_eV':      round(float(e_h2), 5),
            'n_surface':    n_surf,
            'n_atop':       len(atop_sites),
            'n_bridge':     len(bridge_sites),
            'n_hollow':     len(hollow_sites),
            'n_candidate_sites': n_sites,
            'n_skipped':    n_skipped,
            'n_valid_sites': n_valid,
            'dgh_min':      round(float(min(dgh_vals)), 5),
            'dgh_max':      round(float(max(dgh_vals)), 5),
            'dgh_mean':     round(float(np.mean(dgh_vals)), 5),
            'dgh_std':      round(float(np.std(dgh_vals)), 5),
            'sites':        site_results,
        }

        all_results[config_id] = cluster_result

        summary_rows.append({
            'config_id': config_id,
            'size':      size,
            'n_surface': n_surf,
            'n_valid_sites': n_valid,
            'n_skipped': n_skipped,
            'dgh_min':   cluster_result['dgh_min'],
            'dgh_max':   cluster_result['dgh_max'],
            'dgh_mean':  cluster_result['dgh_mean'],
            'dgh_std':   cluster_result['dgh_std'],
        })

        skip_str = f" (skip:{n_skipped})" if n_skipped else ""
        print(f"  {config_id:<20} {n_atoms:>3} {n_surf:>5} {n_valid:>6}{skip_str:<10} "
              f"{min(dgh_vals):>9.3f} {max(dgh_vals):>9.3f} "
              f"{np.mean(dgh_vals):>9.3f} {elapsed:>6.1f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(SUM_JSON, 'w') as f:
        json.dump(summary_rows, f, indent=2)
    write(str(OUT_H_TRAJ), all_h_structs)

    print(f"\nResults saved:")
    print(f"  {OUT_JSON}")
    print(f"  {SUM_JSON}")
    print(f"  {OUT_H_TRAJ}  ({len(all_h_structs)} structures)")

    # ── Size-level summary ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Size-Level ΔGH* Summary (MACE-MP-0)")
    print(f"{'=' * 70}")
    print(f"  {'Size':<8} {'N_clus':>6} {'N_sites':>8} {'ΔGH min':>10} "
          f"{'ΔGH mean':>10} {'ΔGH max':>10}")
    print("  " + "-" * 58)

    from itertools import groupby
    sizes = sorted(set(r['size'] for r in summary_rows))
    for sz in sizes:
        group = [r for r in summary_rows if r['size'] == sz]
        all_dgh_min  = [r['dgh_min']  for r in group]
        all_dgh_mean = [r['dgh_mean'] for r in group]
        all_dgh_max  = [r['dgh_max']  for r in group]
        n_sites_tot  = sum(r['n_valid_sites'] for r in group)
        print(f"  Cu{sz:<6} {len(group):>6} {n_sites_tot:>8} "
              f"{min(all_dgh_min):>10.3f} {np.mean(all_dgh_mean):>10.3f} "
              f"{max(all_dgh_max):>10.3f}")

    print("\nDone. Next step: run 04_gpaw_dgh.py (DFT reference calculations)")
    return all_results, summary_rows


if __name__ == '__main__':
    results, summary = main()
