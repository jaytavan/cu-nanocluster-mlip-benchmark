"""
01_generate_clusters.py

Phase 1 — Pure Cu Nanocluster Generation
-----------------------------------------
Generates pure Cu nanoclusters at 5 target sizes: Cu10, Cu20, Cu30, Cu40, Cu50.

Strategy:
  For each size N:
    1. Build a pool of 100 random compact clusters using FCC sphere + random noise
    2. Apply farthest-point sampling (FPS) in sorted-pairwise-distance space
    3. Select 6 geometrically diverse representatives

FCC sphere approach:
  - Place Cu atoms on an FCC lattice within a sphere sized for N atoms at bulk density
  - Add random noise (±0.3 Å) to break symmetry and produce unique local minima
  - Different noise seeds across the pool give natural geometric diversity

FPS metric: sorted vector of all pairwise distances (no external deps)
            — captures cluster shape robustly for fingerprinting before relaxation

Output:
  results/clusters_pool_CuN.traj    — full random pool per size (optional / debug)
  results/clusters_selected.traj    — 30 FPS-selected clusters (6 × 5 sizes)
  results/generation_summary.json  — pool counts, selected indices, metadata
"""

import json
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────
SIZES       = [10, 20, 30, 40, 50]  # Target Cu cluster sizes (atoms)
N_POOL      = 100                   # Random configs per size before FPS
N_SELECT    = 6                     # FPS selections per size
NOISE_SCALE = 0.30                  # Å — random displacement added to FCC positions
VACUUM      = 10.0                  # Å — padding around cluster
RANDOM_SEED = 42

# Cu FCC lattice constant
CU_A = 3.615  # Å


# ── Cluster generation ──────────────────────────────────────────────────────
def fcc_sphere_cluster(n_atoms: int, noise_scale: float, rng: np.random.Generator) -> Atoms:
    """
    Build a compact Cu cluster:
      1. Generate FCC lattice points within a sphere sized for n_atoms
      2. Sort by radial distance, take innermost n_atoms
      3. Apply random noise to break FCC symmetry
      4. Add vacuum padding, set pbc=False

    The sphere radius is scaled 40% larger than bulk density estimate to ensure
    we always have at least n_atoms available lattice sites to choose from.
    """
    # Estimate radius needed (4 atoms/FCC cell, V_cell = a³)
    # n ≈ (4/a³) × (4π/3)r³  →  r = (3na³/16π)^(1/3) × scale
    r_max = (3 * n_atoms * CU_A**3 / (16 * np.pi)) ** (1 / 3) * 1.4

    # FCC basis in fractional coords
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]])

    n_cells = int(np.ceil(r_max / CU_A)) + 1
    pts = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                for b in basis:
                    pos = (np.array([i, j, k], dtype=float) + b) * CU_A
                    if np.linalg.norm(pos) <= r_max:
                        pts.append(pos)

    pts = np.array(pts)
    # Sort innermost-first so we always pick the most compact n_atoms
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order[:n_atoms]]  # take the n_atoms closest to origin

    # Add random noise to break FCC symmetry
    noise = rng.uniform(-noise_scale, noise_scale, pts.shape)
    pts = pts + noise

    atoms = Atoms(f'Cu{n_atoms}', positions=pts)
    atoms.center(vacuum=VACUUM)
    atoms.set_pbc(False)
    return atoms


# ── Fingerprinting & FPS ────────────────────────────────────────────────────
def pairwise_dist_fingerprint(atoms: Atoms) -> np.ndarray:
    """
    Sorted vector of all pairwise Cu-Cu distances.
    For N atoms: N*(N-1)/2 values, sorted ascending.
    Captures cluster shape without needing a reference frame.
    """
    pos = atoms.get_positions()
    n = len(pos)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(pos[i] - pos[j]))
    return np.sort(dists)


def farthest_point_sampling(fingerprints: list, n_select: int) -> list:
    """
    Greedy farthest-point sampling (FPS) on fingerprint vectors.

    Algorithm:
      1. Start with structure 0 (arbitrary — deterministic for reproducibility)
      2. At each step, pick the structure farthest (in L2) from the already-selected set
      3. Repeat until n_select structures chosen

    Returns list of selected indices into fingerprints.

    Note: all fingerprints for the same N have identical length, so no padding needed.
    """
    n = len(fingerprints)
    fps_arr = np.stack(fingerprints)  # shape (n, d)

    selected = [0]
    min_dists = np.full(n, np.inf)

    for _ in range(n_select - 1):
        last_fp = fps_arr[selected[-1]]
        # Distance from every point to the last-selected
        d_to_last = np.linalg.norm(fps_arr - last_fp, axis=1)
        min_dists = np.minimum(min_dists, d_to_last)
        min_dists[selected] = -np.inf  # exclude already selected

        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return selected


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 1 — Pure Cu Nanocluster Generation")
    print(f"  Sizes:        {SIZES} atoms")
    print(f"  Pool/size:    {N_POOL} random structures")
    print(f"  FPS select:   {N_SELECT} per size")
    print(f"  Noise scale:  ±{NOISE_SCALE} Å on FCC lattice")
    print(f"  Random seed:  {RANDOM_SEED}")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    all_selected = []
    summary = {}

    for n_atoms in SIZES:
        print(f"\n--- Cu{n_atoms} ({n_atoms} atoms) ---")

        # ── Build pool ──────────────────────────────────────────────────────
        pool = []
        for i in range(N_POOL):
            cluster = fcc_sphere_cluster(n_atoms, NOISE_SCALE, rng)
            pool.append(cluster)
        print(f"  Generated pool: {N_POOL} random clusters")

        # ── Fingerprint ─────────────────────────────────────────────────────
        fingerprints = [pairwise_dist_fingerprint(c) for c in pool]
        fp_arr = np.stack(fingerprints)

        # Quick diversity check: std of each fingerprint dimension
        fp_std_mean = float(fp_arr.std(axis=0).mean())
        print(f"  Pool diversity (mean FP std): {fp_std_mean:.3f} Å")

        # ── FPS selection ────────────────────────────────────────────────────
        selected_idx = farthest_point_sampling(fingerprints, N_SELECT)
        selected = [pool[i] for i in selected_idx]

        # ── Tag structures ───────────────────────────────────────────────────
        for i, s in enumerate(selected):
            s.info['size']      = n_atoms
            s.info['formula']   = f"Cu{n_atoms}"
            s.info['config_id'] = f"Cu{n_atoms}_{i:02d}"
            s.info['pool_idx']  = selected_idx[i]

        all_selected.extend(selected)

        # ── Save pool for debug/reference ────────────────────────────────────
        pool_path = RESULTS / f"clusters_pool_Cu{n_atoms}.traj"
        write(str(pool_path), pool)

        print(f"  FPS selected:  {len(selected)} clusters (indices: {selected_idx})")
        print(f"  Pool saved:    {pool_path.name}")
        print(f"  Selected IDs:  {[s.info['config_id'] for s in selected]}")

        # ── Sanity: check no atom overlap in selected clusters ────────────────
        for s in selected:
            pos = s.get_positions()
            dists = []
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dists.append(np.linalg.norm(pos[i] - pos[j]))
            min_d = min(dists)
            if min_d < 1.5:
                print(f"  WARNING: {s.info['config_id']} has atom pair at {min_d:.2f} Å — check!")
        min_ds = []
        for s in selected:
            pos = s.get_positions()
            d_all = []
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    d_all.append(np.linalg.norm(pos[i] - pos[j]))
            min_ds.append(min(d_all))
        print(f"  Min bond dist: {min(min_ds):.2f}–{max(min_ds):.2f} Å  (expect >1.8 Å for FCC+noise)")

        summary[f"Cu{n_atoms}"] = {
            'n_atoms':           n_atoms,
            'pool_size':         N_POOL,
            'n_selected':        len(selected),
            'selected_pool_idx': [int(x) for x in selected_idx],
            'config_ids':        [s.info['config_id'] for s in selected],
            'fp_diversity_std':  round(fp_std_mean, 4),
        }

    # ── Save all selected ────────────────────────────────────────────────────
    selected_path = RESULTS / "clusters_selected.traj"
    write(str(selected_path), all_selected)

    summary['total_selected'] = len(all_selected)
    summary['sizes']          = SIZES
    summary['n_pool']         = N_POOL
    summary['n_select']       = N_SELECT
    summary['noise_scale_A']  = NOISE_SCALE
    summary['random_seed']    = RANDOM_SEED
    summary['cu_lattice_A']   = CU_A

    with open(RESULTS / 'generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Total clusters selected: {len(all_selected)}")
    print(f"  → {selected_path}")
    print(f"  → results/generation_summary.json")
    print(f"\nNext: run 02_mace_relax.py")

    return all_selected, summary


if __name__ == '__main__':
    structures, summary = main()
