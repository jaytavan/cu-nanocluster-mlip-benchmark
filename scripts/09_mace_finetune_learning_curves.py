"""
Script 09 — MACE Fine-tuning + Learning Curves
================================================
Phase 3 of the mlip-nanocluster-benchmark publication.

Goal: Measure how quickly MACE-MP-0 fine-tuned on GPAW ΔGH* data recovers
accuracy compared to the zero-shot foundation model baseline.

Approach:
- Collect all GPAW-computed cluster+H structures and bare cluster structures
- Fine-tune MACE-MP-0 (medium) at N = 10, 20, 40, 60, 80 training points
- Hold out 20% for testing (stratified by cluster size)
- 3 random seeds per N → mean ± std learning curve
- Evaluate: MAE on ΔGH* (eV) vs GPAW reference
- Generates fig8_learning_curves.png

v2 data (PBE+D3(BJ), forces stored):
- If forces_eVA present in site results, trains with energy + forces (forces_weight=1.0)
- If forces absent (v1 legacy data), falls back to energy-only (forces_weight=0.0)
- Bare-cluster forces loaded from cluster_energy_{CID}_d3.json

Run after GPAW v2 re-run completes:
    python3 scripts/09_mace_finetune_learning_curves.py

Prerequisites:
    results/gpaw/{Cu10,Cu20,Cu30,Cu40,Cu50}/results_Cu*.json  (v2 with forces)
    results/gpaw/sites_prepared.json  (MACE-relaxed H positions)
    ~/.cache/mace/20231203mace128L1_epoch199model  (MACE-MP-0 medium, auto-downloaded)
"""

import json
import subprocess
import shutil
import tempfile
import time
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
RESULTS_DIR  = PROJECT_DIR / "results"
GPAW_DIR     = RESULTS_DIR / "gpaw"
VIZ_DIR      = PROJECT_DIR / "viz"
FINETUNE_DIR = RESULTS_DIR / "finetune"

MACE_MODEL_PATH = Path.home() / ".cache/mace/20231203mace128L1_epoch199model"
MACE_PYTHON     = sys.executable  # same conda env
ZPE_CORR        = 0.24  # eV — standard CHE correction for H*

# Learning curve sizes and random seeds
TRAIN_SIZES = [10, 20, 40, 60, 80]
N_SEEDS     = 3
TEST_FRAC   = 0.20  # hold-out fraction


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_gpaw_sites() -> List[Dict]:
    """Load all completed GPAW site results across cluster sizes."""
    all_sites = []
    for sz in [10, 20, 30, 40, 50]:
        result_file = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_00.json"
        if not result_file.exists():
            print(f"  [skip] Cu{sz}: results file not found ({result_file})")
            continue
        data = json.load(open(result_file))
        n_sites = sum(1 for s in data["sites"] if s["status"] == "ok")
        print(f"  Cu{sz}: {n_sites} sites loaded (MAE={data['mae_eV']:.3f} eV)")
        all_sites.extend([s for s in data["sites"] if s["status"] == "ok"])
    return all_sites


def load_prepared_sites() -> Dict[str, Dict]:
    """Load MACE-relaxed cluster+H positions keyed by site_global_id."""
    prep = json.load(open(GPAW_DIR / "sites_prepared.json"))
    return {s["site_global_id"]: s for s in prep}


def load_bare_cluster_structures() -> Dict[str, "ase.Atoms"]:
    """Load MACE-relaxed bare cluster structures keyed by config_id."""
    from ase.io import read
    traj = read(str(RESULTS_DIR / "clusters_relaxed.traj"), index=":")
    # Each config_id is Cu{N}_00 — pick the representative cluster (index 0 per size)
    clusters = {}
    for atoms in traj:
        n = len(atoms)
        cid = f"Cu{n}_00"
        if cid not in clusters:
            clusters[cid] = atoms
    return clusters


def load_bare_cluster_forces() -> Dict[str, List]:
    """
    Load GPAW+D3 forces for bare clusters from cluster_energy_{CID}_d3.json files.
    Returns dict keyed by config_id (e.g. 'Cu50_00') → forces list shape (N,3).
    Returns empty dict if files not found (v1 legacy data, no forces).
    """
    forces_map = {}
    for sz in [10, 20, 30, 40, 50]:
        cid = f"Cu{sz}_00"
        cache_file = GPAW_DIR / f"Cu{sz}" / f"cluster_energy_{cid}_d3.json"
        if cache_file.exists():
            data = json.load(open(cache_file))
            if "forces_eVA" in data and data["forces_eVA"] is not None:
                forces_map[cid] = data["forces_eVA"]
    if forces_map:
        print(f"  Bare cluster forces loaded for: {list(forces_map.keys())}")
    else:
        print("  [warn] No bare cluster force files found — energy-only training")
    return forces_map


# ── Extended XYZ Writer ────────────────────────────────────────────────────────

def write_extxyz(filepath: Path, structures: List[Tuple]) -> Tuple[int, bool]:
    """
    Write extended XYZ file for MACE training.

    structures: list of (atoms, energy_eV, config_type_str)
                      or (atoms, energy_eV, config_type_str, forces)
      forces: array-like shape (N,3) in eV/Å, or None to omit.

    Returns: (n_written, has_forces) — has_forces=True if any structure had forces.
    """
    written = 0
    any_forces = False
    with open(filepath, "w") as f:
        for entry in structures:
            atoms, energy, config_type = entry[0], entry[1], entry[2]
            forces = entry[3] if len(entry) == 4 else None

            n = len(atoms)
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()

            # Lattice string (3×3 cell flattened row-major)
            lattice_str = " ".join(f"{v:.6f}" for v in cell.flatten())
            pbc_str = "T T T" if all(pbc) else "F F F"

            if forces is not None:
                props = "Properties=species:S:1:pos:R:3:forces:R:3"
                any_forces = True
            else:
                props = "Properties=species:S:1:pos:R:3"

            comment = (
                f'Lattice="{lattice_str}" '
                f'{props} '
                f'energy={energy:.8f} '
                f'config_type={config_type} '
                f'pbc="{pbc_str}"'
            )
            f.write(f"{n}\n{comment}\n")
            if forces is not None:
                forces_arr = np.array(forces)
                for sym, pos, frc in zip(symbols, positions, forces_arr):
                    f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                            f"{frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n")
            else:
                for sym, pos in zip(symbols, positions):
                    f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            written += 1
    return written, any_forces


# ── Structure Reconstruction ───────────────────────────────────────────────────

def site_to_cluster_h_atoms(gpaw_site: Dict, prepared: Dict) -> "ase.Atoms":
    """Reconstruct cluster+H ASE Atoms from GPAW site record + prepared positions."""
    from ase import Atoms
    import numpy as np

    sid = gpaw_site["site_global_id"]
    if sid not in prepared:
        return None

    prep = prepared[sid]
    n_cu = gpaw_site["size"]
    cluster_pos = np.array(prep["cluster_pos"])
    h_pos = np.array(prep["h_pos"])
    cell = np.array(prep["cell"])

    symbols = ["Cu"] * n_cu + ["H"]
    positions = np.vstack([cluster_pos, h_pos.reshape(1, 3)])

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
    atoms.center(vacuum=10.0)
    atoms.set_pbc(True)  # MACE needs periodic cell
    return atoms


def bare_cluster_atoms(cluster: "ase.Atoms") -> "ase.Atoms":
    """Return bare cluster with large vacuum cell for MACE."""
    atoms = cluster.copy()
    atoms.center(vacuum=10.0)
    atoms.set_pbc(True)
    return atoms


# ── MACE Fine-tuning ──────────────────────────────────────────────────────────

def finetune_mace(
    train_xyz: Path,
    valid_xyz: Path,
    output_dir: Path,
    run_name: str,
    has_forces: bool = False,
    max_epochs: int = 300,
    lr: float = 1e-4,
    batch_size: int = 4,
) -> Path:
    """
    Fine-tune MACE-MP-0 (medium) on training data.

    has_forces: if True, trains jointly on energy + forces (forces_weight=1.0).
                if False, energy-only (forces_weight=0.0, legacy v1 data).

    Returns path to the best model checkpoint, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    forces_weight = "1.0" if has_forces else "0.0"

    # E0s: use MACE-MP-0 default (average over training structures)
    cmd = [
        MACE_PYTHON, "-m", "mace.cli.run_train",
        f"--name={run_name}",
        f"--train_file={train_xyz}",
        f"--valid_file={valid_xyz}",
        f"--model_dir={output_dir}",
        f"--results_dir={output_dir}",
        f"--checkpoints_dir={output_dir}",
        f"--log_dir={output_dir}",
        "--foundation_model=medium",
        "--multiheads_finetuning=False",
        f"--max_num_epochs={max_epochs}",
        f"--batch_size={batch_size}",
        f"--valid_batch_size={batch_size}",
        f"--lr={lr}",
        "--energy_weight=1.0",
        f"--forces_weight={forces_weight}",
        "--loss=weighted",
        "--E0s=average",
        "--energy_key=energy",
        "--model=MACE",
        "--r_max=6.0",
        "--num_radial_basis=8",
        "--num_cutoff_basis=5",
        "--max_ell=3",
        "--num_interactions=2",
        "--correlation=3",
        "--num_channels=128",
        "--hidden_irreps=128x0e+128x1o",
        "--patience=50",
        "--device=cuda",
        "--default_dtype=float32",
        "--seed=42",
        "--keep_checkpoints",
        "--keep_isolated_atoms=True",
        "--save_cpu",
    ]
    if has_forces:
        cmd.append("--forces_key=forces")

    log_file = output_dir / f"{run_name}.log"
    t0 = time.time()

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
            timeout=3600,  # 1 hr max per fine-tune run
        )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    [FAIL] {run_name} — returncode={result.returncode} ({elapsed:.0f}s)")
        # Print last 10 lines of log for debugging
        lines = log_file.read_text().strip().split("\n")
        for line in lines[-10:]:
            print(f"      {line}")
        return None

    # Find the best model checkpoint
    best_model = output_dir / f"{run_name}_best_val_inf.model"
    if not best_model.exists():
        # Try alternative naming
        models = sorted(output_dir.glob(f"{run_name}*.model"))
        if models:
            best_model = models[-1]
        else:
            print(f"    [FAIL] No model file found for {run_name}")
            return None

    print(f"    [OK] {run_name} — {elapsed:.0f}s → {best_model.name}")
    return best_model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_finetuned(
    model_path: Path,
    test_sites: List[Dict],
    prepared: Dict,
    bare_clusters: Dict,
    e_h2: float,
) -> Dict:
    """
    Evaluate fine-tuned MACE on test sites.

    Returns dict with MAE, RMSE, Pearson r, and per-site predictions.
    """
    from mace.calculators import MACECalculator
    import numpy as np

    calc = MACECalculator(
        model_paths=[str(model_path)],
        device="cuda",
        default_dtype="float32",
    )

    predictions = []
    for site in test_sites:
        cid = site["config_id"]
        n_cu = site["size"]

        # Bare cluster energy
        cluster = bare_clusters.get(cid)
        if cluster is None:
            continue
        cluster_calc = bare_cluster_atoms(cluster)
        cluster_calc.calc = calc
        try:
            e_cluster_ft = cluster_calc.get_potential_energy()
        except Exception as e:
            print(f"    [skip] bare cluster {cid}: {e}")
            continue

        # Cluster+H energy
        cluster_h = site_to_cluster_h_atoms(site, prepared)
        if cluster_h is None:
            continue
        cluster_h.calc = calc
        try:
            e_clus_h_ft = cluster_h.get_potential_energy()
        except Exception as e:
            print(f"    [skip] cluster+H {site['site_global_id']}: {e}")
            continue

        dgh_ft = e_clus_h_ft - e_cluster_ft - 0.5 * e_h2 + ZPE_CORR
        dgh_gpaw = site["dgh_gpaw_eV"]

        predictions.append({
            "site_global_id": site["site_global_id"],
            "size": n_cu,
            "dgh_ft_eV": float(dgh_ft),
            "dgh_gpaw_eV": float(dgh_gpaw),
            "error_eV": float(dgh_ft - dgh_gpaw),
        })

    if not predictions:
        return {"mae": None, "rmse": None, "pearson_r": None, "n": 0}

    errors = np.array([p["error_eV"] for p in predictions])
    gpaw_vals = np.array([p["dgh_gpaw_eV"] for p in predictions])
    ft_vals = np.array([p["dgh_ft_eV"] for p in predictions])

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    pearson_r = float(np.corrcoef(gpaw_vals, ft_vals)[0, 1]) if len(predictions) > 1 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "n": len(predictions),
        "predictions": predictions,
    }


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_learning_curves(lc_results: Dict, zero_shot: Dict, output_path: Path):
    """
    Generate Fig 8: Learning curves (MAE vs N_train).

    lc_results: {n_train: {"mae_mean": float, "mae_std": float, ...}} per model
    zero_shot: {"MACE-MP-0": mae, "CHGNet": mae, "TensorNet": mae}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Learning Curves: Fine-tuned MACE-MP-0 vs Foundation Models\n"
        r"Cu$_{10}$–Cu$_{50}$ Nanoclusters, H* Adsorption (ΔG$_H^*$)",
        fontsize=13, fontweight="bold",
    )

    # Colors
    ft_color   = "#e74c3c"
    mace_color = "#3498db"
    chg_color  = "#2ecc71"
    tn_color   = "#9b59b6"

    for ax_idx, (ax, metric) in enumerate(zip(axes, ["mae", "pearson_r"])):
        ylabel = "MAE (eV)" if metric == "mae" else "Pearson r"
        title  = "Accuracy (MAE)" if metric == "mae" else "Rank-order fidelity (Pearson r)"

        # Fine-tuned MACE learning curve
        if lc_results:
            ns   = sorted(lc_results.keys())
            means = [lc_results[n][f"{metric}_mean"] for n in ns]
            stds  = [lc_results[n][f"{metric}_std"]  for n in ns]
            ax.plot(ns, means, "o-", color=ft_color, lw=2, ms=7,
                    label="MACE-MP-0 fine-tuned", zorder=3)
            ax.fill_between(ns,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=ft_color)

        # Zero-shot baselines
        x_range = [min(TRAIN_SIZES) * 0.8, max(TRAIN_SIZES) * 1.1]
        ax.axhline(zero_shot.get("MACE-MP-0", {}).get(metric, np.nan),
                   ls="--", color=mace_color, lw=1.5, label="MACE-MP-0 zero-shot", zorder=2)
        ax.axhline(zero_shot.get("CHGNet", {}).get(metric, np.nan),
                   ls="-.", color=chg_color,  lw=1.5, label="CHGNet zero-shot", zorder=2)
        ax.axhline(zero_shot.get("TensorNet", {}).get(metric, np.nan),
                   ls=":",  color=tn_color,   lw=1.5, label="TensorNet zero-shot", zorder=2)

        ax.set_xlabel("Training set size N (DFT points)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)

        if metric == "mae":
            ax.set_ylim(bottom=0)

        ax.set_xticks(TRAIN_SIZES)
        ax.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import random

    VIZ_DIR.mkdir(exist_ok=True)
    FINETUNE_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 09 — MACE Fine-tuning + Learning Curves")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading GPAW data...")
    gpaw_sites = load_all_gpaw_sites()
    print(f"  Total GPAW sites: {len(gpaw_sites)}")

    if len(gpaw_sites) < 20:
        print("  ERROR: Not enough GPAW data for learning curves. Run after all sizes complete.")
        sys.exit(1)

    prepared      = load_prepared_sites()
    bare_clusters = load_bare_cluster_structures()
    bare_forces   = load_bare_cluster_forces()   # {} if v1 (no forces)

    # Detect whether forces are available (v2 data)
    first_with_forces = next(
        (s for s in gpaw_sites if s.get("forces_eVA") is not None), None
    )
    USE_FORCES = (first_with_forces is not None) and bool(bare_forces)
    print(f"  Force training: {'ENABLED (v2 data)' if USE_FORCES else 'DISABLED (energy-only / v1 data)'}")

    # Extract H2 reference energy from any site
    e_h2 = gpaw_sites[0]["e_h2_eV"]
    print(f"  H2 reference: {e_h2:.5f} eV")

    # ── 2. Build train/test split ─────────────────────────────────────────
    print("\n[2] Building stratified train/test split...")
    rng = np.random.default_rng(seed=42)

    # Group sites by cluster size
    by_size = {}
    for site in gpaw_sites:
        sz = site["size"]
        by_size.setdefault(sz, []).append(site)

    test_sites  = []
    train_pool  = []
    for sz, sites in sorted(by_size.items()):
        n_test = max(1, round(len(sites) * TEST_FRAC))
        idx = rng.permutation(len(sites))
        test_sites.extend([sites[i] for i in idx[:n_test]])
        train_pool.extend([sites[i] for i in idx[n_test:]])

    print(f"  Train pool: {len(train_pool)} sites | Test: {len(test_sites)} sites")

    # ── 3. Load zero-shot baseline from benchmark_summary.json ────────────
    print("\n[3] Loading zero-shot baselines...")
    zero_shot = {}
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if summary_path.exists():
        bsum = json.load(open(summary_path))
        for model_name, mdata in bsum.items():
            if isinstance(mdata, dict) and "mae_eV" in mdata:
                zero_shot[model_name] = {
                    "mae": mdata["mae_eV"],
                    "pearson_r": mdata.get("pearson_r", 0.0),
                }
        print(f"  Loaded: {list(zero_shot.keys())}")
    else:
        print("  [warn] benchmark_summary.json not found — baselines will be empty")
        print("         Run script 07 first to generate it.")

    # ── 4. Prepare valid set XYZ (fixed across all runs) ─────────────────
    print("\n[4] Preparing validation + test XYZ structures...")

    valid_structures = []
    for site in test_sites:
        atoms = site_to_cluster_h_atoms(site, prepared)
        if atoms is None:
            continue
        energy = site["e_clus_h_eV"]
        f_clus_h = site.get("forces_eVA") if USE_FORCES else None
        valid_structures.append((atoms, energy, f"Cu{site['size']}_H", f_clus_h))

        # Include bare cluster in valid (to anchor absolute energies)
        cid = site["config_id"]
        if cid in bare_clusters:
            bare = bare_cluster_atoms(bare_clusters[cid])
            e_bare = site["e_cluster_eV"]
            f_bare = bare_forces.get(cid) if USE_FORCES else None
            valid_structures.append((bare, e_bare, f"Cu{site['size']}_bare", f_bare))

    valid_xyz = FINETUNE_DIR / "valid.xyz"
    n_valid, _ = write_extxyz(valid_xyz, valid_structures)
    print(f"  Written: {valid_xyz} ({n_valid} structures)")

    # ── 5. Learning curve loop ────────────────────────────────────────────
    print("\n[5] Running fine-tuning learning curves...")
    lc_results = {}
    lc_raw = {}  # {n_train: [{"mae": ..., "rmse": ..., "pearson_r": ...}, ...]}

    for n_train in TRAIN_SIZES:
        if n_train > len(train_pool):
            print(f"\n  [skip] N={n_train}: not enough training data ({len(train_pool)} in pool)")
            continue

        print(f"\n  ── N={n_train} training points ──")
        seed_results = []

        for seed in range(N_SEEDS):
            rng_seed = np.random.default_rng(seed=seed * 100 + n_train)
            # Stratified sample: pick n_train from pool, preserving size distribution
            sampled = []
            remaining = n_train
            for sz in sorted(by_size.keys()):
                pool_sz = [s for s in train_pool if s["size"] == sz]
                n_from_sz = max(1, round(n_train * len(pool_sz) / len(train_pool)))
                n_from_sz = min(n_from_sz, len(pool_sz), remaining)
                if n_from_sz == 0:
                    continue
                idx = rng_seed.choice(len(pool_sz), size=n_from_sz, replace=False)
                sampled.extend([pool_sz[i] for i in idx])
                remaining -= n_from_sz

            # Build training structures
            train_structures = []
            for site in sampled:
                atoms = site_to_cluster_h_atoms(site, prepared)
                if atoms is None:
                    continue
                f_clus_h = site.get("forces_eVA") if USE_FORCES else None
                train_structures.append(
                    (atoms, site["e_clus_h_eV"], f"Cu{site['size']}_H", f_clus_h)
                )
                cid = site["config_id"]
                if cid in bare_clusters:
                    bare = bare_cluster_atoms(bare_clusters[cid])
                    f_bare = bare_forces.get(cid) if USE_FORCES else None
                    train_structures.append(
                        (bare, site["e_cluster_eV"], f"Cu{site['size']}_bare", f_bare)
                    )

            run_name = f"mace_ft_N{n_train:03d}_seed{seed}"
            run_dir  = FINETUNE_DIR / run_name
            train_xyz = run_dir / "train.xyz"
            run_dir.mkdir(exist_ok=True)
            n_written, train_has_forces = write_extxyz(train_xyz, train_structures)
            print(f"    seed={seed}: {n_written} structures "
                  f"({'E+F' if train_has_forces else 'E-only'}) → fine-tuning...")

            # Fine-tune
            model_path = finetune_mace(
                train_xyz=train_xyz,
                valid_xyz=valid_xyz,
                output_dir=run_dir,
                run_name=run_name,
                has_forces=train_has_forces,
                max_epochs=300,
                lr=5e-5,
                batch_size=min(4, n_written),
            )

            if model_path is None:
                print(f"    [WARN] Fine-tuning failed for seed={seed}, skipping.")
                continue

            # Evaluate
            print(f"    Evaluating on {len(test_sites)} test sites...")
            eval_result = evaluate_finetuned(
                model_path=model_path,
                test_sites=test_sites,
                prepared=prepared,
                bare_clusters=bare_clusters,
                e_h2=e_h2,
            )
            print(f"    MAE={eval_result['mae']:.4f} eV | "
                  f"RMSE={eval_result['rmse']:.4f} eV | "
                  f"r={eval_result['pearson_r']:.3f} (n={eval_result['n']})")
            seed_results.append(eval_result)

            # Save per-run result
            run_result_path = run_dir / "eval_result.json"
            json.dump(eval_result, open(run_result_path, "w"), indent=2)

        if not seed_results:
            print(f"  [WARN] No successful runs for N={n_train}")
            continue

        # Aggregate across seeds
        valid_results = [r for r in seed_results if r["mae"] is not None]
        if not valid_results:
            continue

        lc_raw[n_train] = valid_results
        lc_results[n_train] = {
            "mae_mean":       float(np.mean([r["mae"] for r in valid_results])),
            "mae_std":        float(np.std([r["mae"]  for r in valid_results])),
            "rmse_mean":      float(np.mean([r["rmse"] for r in valid_results])),
            "rmse_std":       float(np.std([r["rmse"]  for r in valid_results])),
            "pearson_r_mean": float(np.mean([r["pearson_r"] for r in valid_results])),
            "pearson_r_std":  float(np.std([r["pearson_r"]  for r in valid_results])),
            "n_seeds_ok":     len(valid_results),
        }
        print(f"\n  N={n_train} summary: "
              f"MAE={lc_results[n_train]['mae_mean']:.3f}±{lc_results[n_train]['mae_std']:.3f} eV")

    # ── 6. Save learning curve data ───────────────────────────────────────
    print("\n[6] Saving learning curve results...")
    lc_path = RESULTS_DIR / "learning_curves.json"
    json.dump({
        "train_sizes": TRAIN_SIZES,
        "n_seeds":     N_SEEDS,
        "test_frac":   TEST_FRAC,
        "n_test":      len(test_sites),
        "zero_shot":   zero_shot,
        "learning_curve": lc_results,
    }, open(lc_path, "w"), indent=2)
    print(f"  Saved: {lc_path}")

    # ── 7. Generate Fig 8 ─────────────────────────────────────────────────
    print("\n[7] Generating Fig 8 — Learning Curves...")
    # Convert zero_shot metric keys for plotting
    zs_plot = {}
    for model_name, metrics in zero_shot.items():
        zs_plot[model_name] = {
            "mae": metrics.get("mae", np.nan),
            "pearson_r": metrics.get("pearson_r", np.nan),
        }

    # Pass lc_results directly — plot function expects {metric}_mean / {metric}_std keys
    lc_for_plot = lc_results

    fig8_path = VIZ_DIR / "fig8_learning_curves.png"
    plot_learning_curves(lc_for_plot, zs_plot, fig8_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("LEARNING CURVE SUMMARY")
    print("=" * 65)
    print(f"\nZero-shot baselines (vs GPAW):")
    for model, m in zero_shot.items():
        print(f"  {model:15s}: MAE={m.get('mae', 'n/a'):.3f} eV, r={m.get('pearson_r', 'n/a'):.3f}")

    if lc_results:
        print(f"\nFine-tuned MACE MAE at each N:")
        for n, v in sorted(lc_results.items()):
            print(f"  N={n:3d}: {v['mae_mean']:.3f} ± {v['mae_std']:.3f} eV "
                  f"(r={v['pearson_r_mean']:.3f} ± {v['pearson_r_std']:.3f})")

        best_n = min(lc_results, key=lambda n: lc_results[n]["mae_mean"])
        best_mae = lc_results[best_n]["mae_mean"]
        zs_mace_mae = zero_shot.get("MACE-MP-0", {}).get("mae", None)
        if zs_mace_mae:
            improvement = (zs_mace_mae - best_mae) / zs_mace_mae * 100
            print(f"\n  Best: N={best_n} → MAE={best_mae:.3f} eV "
                  f"({improvement:.0f}% improvement over zero-shot MACE)")

    print(f"\nFig 8: {fig8_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
