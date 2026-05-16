"""
02_mace_relax.py

Phase 1 — MACE-MP-0 Cluster Relaxation
----------------------------------------
Relaxes all 30 FPS-selected pure Cu nanoclusters (Cu10–Cu50) using
MACE-MP-0 medium universal potential on GPU.

The MACE relaxation serves two purposes:
  1. Drives random FCC-sphere initial geometries to proper local energy minima
  2. Produces the reference geometry used by ALL downstream scripts
     (both other MLIPs and GPAW DFT use the MACE-relaxed geometry)

Using a single reference geometry avoids confounding geometry differences
with potential energy differences — critical for a clean benchmark.

Force convergence: 0.05 eV/Å  (standard for nanocluster relaxation)
Device: CUDA (RTX 4060)

Input:  results/clusters_selected.traj  (30 structures from 01_generate_clusters.py)
Output: results/clusters_relaxed.traj   — MACE-relaxed structures
        results/relaxation_log.json     — energy, fmax, convergence per structure
"""

import json
import time
import numpy as np
from pathlib import Path

from ase.io import read, write
from ase.optimize import LBFGS

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).parent.parent
RESULTS  = PROJECT / "results"
IN_TRAJ  = RESULTS / "clusters_selected.traj"
OUT_TRAJ = RESULTS / "clusters_relaxed.traj"
LOG_FILE = RESULTS / "relaxation_log.json"

# ── Parameters ─────────────────────────────────────────────────────────────
FMAX       = 0.05   # eV/Å — standard convergence criterion
MAX_STEPS  = 500
DEVICE     = "cuda"
MACE_MODEL = "medium"


# ── GPU check & MACE loader ─────────────────────────────────────────────────
def load_mace():
    """Load MACE-MP-0. Raise loudly if GPU not available — never run on CPU."""
    import torch
    from mace.calculators import mace_mp

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Do not proceed with CPU-only MACE relaxation.\n"
            "Verify with: python3 -c \"import torch; print(torch.cuda.is_available())\""
        )
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print(f"  Model: MACE-MP-0 ({MACE_MODEL}), dtype=float32")
    print(f"  Loading...")

    calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float32")
    print("  MACE calculator ready.\n")
    return calc


# ── Single cluster relaxation ───────────────────────────────────────────────
def relax_cluster(atoms, calc, config_id: str) -> tuple:
    """
    Relax a single Cu cluster with LBFGS.

    Returns (relaxed_atoms, log_dict).
    The relaxed atoms retain all .info metadata from the input.
    """
    atoms = atoms.copy()
    atoms.set_pbc(False)
    atoms.calc = calc

    e_before = atoms.get_potential_energy()

    t0 = time.time()
    opt = LBFGS(atoms, logfile=None)
    converged = opt.run(fmax=FMAX, steps=MAX_STEPS)
    elapsed = time.time() - t0

    e_after    = atoms.get_potential_energy()
    forces     = atoms.get_forces()
    fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))

    # Compute cluster diameter (max pairwise distance)
    pos    = atoms.get_positions()
    center = pos.mean(axis=0)
    radii  = np.linalg.norm(pos - center, axis=1)
    diameter = float(2 * radii.max())

    log = {
        'config_id':   config_id,
        'formula':     atoms.get_chemical_formula(),
        'n_atoms':     len(atoms),
        'size':        atoms.info.get('size', -1),
        'e_before_eV': round(float(e_before), 5),
        'e_after_eV':  round(float(e_after), 5),
        'delta_e_eV':  round(float(e_after - e_before), 5),
        'fmax_eV_A':   round(fmax_final, 5),
        'converged':   bool(converged),
        'n_steps':     opt.get_number_of_steps(),
        'diameter_A':  round(diameter, 3),
        'time_s':      round(elapsed, 2),
    }

    # Write key results back into atoms.info for downstream scripts
    atoms.info['config_id']  = config_id
    atoms.info['e_relaxed']  = round(float(e_after), 5)
    atoms.info['converged']  = bool(converged)
    atoms.info['diameter_A'] = round(diameter, 3)

    return atoms, log


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase 1 — MACE-MP-0 Cluster Relaxation")
    print(f"  Input:     {IN_TRAJ}")
    print(f"  fmax:      {FMAX} eV/Å")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Device:    {DEVICE}")
    print("=" * 65)

    if not IN_TRAJ.exists():
        raise FileNotFoundError(
            f"{IN_TRAJ} not found — run 01_generate_clusters.py first"
        )

    structures = read(str(IN_TRAJ), index=':')
    print(f"\nLoaded {len(structures)} clusters from {IN_TRAJ.name}")

    # Group by size for reporting
    sizes = sorted(set(a.info.get('size', len(a)) for a in structures))
    for sz in sizes:
        group = [a for a in structures if a.info.get('size', len(a)) == sz]
        print(f"  Cu{sz}: {len(group)} structures")

    print()
    calc = load_mace()

    # ── Relaxation loop ──────────────────────────────────────────────────────
    header = f"  {'ID':<20} {'N':>4} {'E_before':>10} {'E_after':>10} {'ΔE':>8} {'fmax':>7} {'diam':>6} {'C':>2} {'t(s)':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    relaxed  = []
    logs     = []
    n_failed = 0

    for i, atoms in enumerate(structures):
        config_id = atoms.info.get('config_id', f"struct_{i:03d}")

        try:
            rel_atoms, log = relax_cluster(atoms, calc, config_id)
            relaxed.append(rel_atoms)
            logs.append(log)

            c_str = "✓" if log['converged'] else "~"
            print(f"  {config_id:<20} {log['n_atoms']:>4} "
                  f"{log['e_before_eV']:>10.3f} {log['e_after_eV']:>10.3f} "
                  f"{log['delta_e_eV']:>8.3f} {log['fmax_eV_A']:>7.4f} "
                  f"{log['diameter_A']:>6.2f} {c_str:>2} {log['time_s']:>5.1f}")

        except Exception as e:
            print(f"  {config_id:<20}  FAILED: {e}")
            n_failed += 1
            logs.append({'config_id': config_id, 'error': str(e)})

    # ── Save outputs ─────────────────────────────────────────────────────────
    write(str(OUT_TRAJ), relaxed)
    print(f"\nRelaxed structures saved: {len(relaxed)}  → {OUT_TRAJ}")

    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"Relaxation log saved:     {LOG_FILE}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Relaxation Summary by Size")
    print(f"{'=' * 65}")
    print(f"  {'Size':<8} {'N':>4} {'Converged':>10} {'E_mean (eV)':>13} {'Diam_mean (Å)':>14}")
    print("  " + "-" * 55)

    for sz in sizes:
        group_logs = [l for l in logs if l.get('size') == sz and 'e_after_eV' in l]
        if not group_logs:
            continue
        n_conv  = sum(1 for l in group_logs if l.get('converged'))
        e_vals  = [l['e_after_eV'] for l in group_logs]
        d_vals  = [l['diameter_A'] for l in group_logs]
        print(f"  Cu{sz:<6} {sz:>4} {n_conv}/{len(group_logs):>8} "
              f"{np.mean(e_vals):>13.3f} {np.mean(d_vals):>14.3f}")

    n_conv_total = sum(1 for l in logs if l.get('converged'))
    n_failed_conv = sum(1 for l in logs if 'converged' in l and not l['converged'])
    print(f"\n  Total: {len(structures)} structures")
    print(f"  Converged:    {n_conv_total}")
    print(f"  Not conv (~): {n_failed_conv}  (usable — check fmax in log)")
    print(f"  Errors:       {n_failed}")

    print("\nDone. Next step: run 03_adsorption_sites.py")
    return relaxed, logs


if __name__ == '__main__':
    relaxed, logs = main()
