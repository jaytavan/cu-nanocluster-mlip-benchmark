"""
Script 13 — CHGNet Fine-tuning + Learning Curves
=================================================
Phase 3b of the mlip-nanocluster-benchmark publication.

Goal: Measure how quickly CHGNet fine-tuned on GPAW ΔGH* data recovers
accuracy compared to the zero-shot CHGNet baseline, and compare against
fine-tuned MACE-MP-0 from script 09.

Approach:
- Same train/test split as script 09 (seed=42, stratified by size, 20% test)
- Fine-tune CHGNet at N = 10, 20, 40, 60, 80 training points
- 3 random seeds per N → mean ± std learning curve
- Train on energy + forces (GPAW v2 data)
- Evaluate: MAE on ΔGH* (eV) vs GPAW reference
- Updates fig8_learning_curves.png with CHGNet fine-tuning curve added

Run after Script 09 completes:
    python3 scripts/13_chgnet_finetune_lc.py

Prerequisites:
    results/gpaw/{Cu10,Cu20,Cu30,Cu40,Cu50}/results_Cu*.json  (v2 with forces)
    results/gpaw/sites_prepared.json
    results/learning_curves.json  (script 09 MACE fine-tuning results)
    results/benchmark_summary.json  (zero-shot baselines)
"""

import json
import os
import time
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
PROJECT_DIR   = SCRIPT_DIR.parent
RESULTS_DIR   = PROJECT_DIR / "results"
GPAW_DIR      = RESULTS_DIR / "gpaw"
VIZ_DIR       = PROJECT_DIR / "viz"
CHGNET_FT_DIR = RESULTS_DIR / "chgnet_ft"

ZPE_CORR   = 0.24   # eV — standard CHE correction for H*
TRAIN_SIZES = [10, 20, 40, 60, 80]
N_SEEDS     = 3
TEST_FRAC   = 0.20
EPOCHS      = 50
LR          = 1e-3


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_all_gpaw_sites() -> List[Dict]:
    all_sites = []
    for sz in [10, 20, 30, 40, 50]:
        result_file = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_00.json"
        if not result_file.exists():
            print(f"  [skip] Cu{sz}: results file not found")
            continue
        data = json.load(open(result_file))
        sites = [s for s in data["sites"] if s["status"] == "ok"]
        print(f"  Cu{sz}: {len(sites)} sites (MAE={data['mae_eV']:.3f} eV)")
        all_sites.extend(sites)
    return all_sites


def load_prepared_sites() -> Dict[str, Dict]:
    prep = json.load(open(GPAW_DIR / "sites_prepared.json"))
    return {s["site_global_id"]: s for s in prep}


def load_bare_cluster_forces() -> Dict[str, List]:
    forces_map = {}
    for sz in [10, 20, 30, 40, 50]:
        cid = f"Cu{sz}_00"
        cache_file = GPAW_DIR / f"Cu{sz}" / f"cluster_energy_{cid}_d3.json"
        if cache_file.exists():
            data = json.load(open(cache_file))
            if data.get("forces_eVA") is not None:
                forces_map[cid] = data["forces_eVA"]
    print(f"  Bare cluster forces loaded for: {list(forces_map.keys())}")
    return forces_map


# ── ASE → pymatgen conversion ──────────────────────────────────────────────────

def site_to_cluster_h_atoms(gpaw_site: Dict, prepared: Dict):
    from ase import Atoms
    sid = gpaw_site["site_global_id"]
    if sid not in prepared:
        return None
    prep = prepared[sid]
    n_cu = gpaw_site["size"]
    cluster_pos = np.array(prep["cluster_pos"])
    h_pos = np.array(prep["h_pos"])
    cell = np.array(prep["cell"])
    positions = np.vstack([cluster_pos, h_pos.reshape(1, 3)])
    atoms = Atoms(symbols=["Cu"] * n_cu + ["H"], positions=positions, cell=cell)
    atoms.center(vacuum=10.0)
    atoms.set_pbc(True)
    return atoms


def bare_cluster_atoms_from_traj(bare_clusters: Dict, cid: str):
    cluster = bare_clusters.get(cid)
    if cluster is None:
        return None
    atoms = cluster.copy()
    atoms.center(vacuum=10.0)
    atoms.set_pbc(True)
    return atoms


def load_bare_cluster_structures() -> Dict[str, "ase.Atoms"]:
    from ase.io import read
    traj = read(str(RESULTS_DIR / "clusters_relaxed.traj"), index=":")
    clusters = {}
    for atoms in traj:
        cid = f"Cu{len(atoms)}_00"
        if cid not in clusters:
            clusters[cid] = atoms
    return clusters


def ase_to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


# ── CHGNet Training ────────────────────────────────────────────────────────────

def build_chgnet_dataset(
    sites: List[Dict],
    prepared: Dict,
    bare_clusters: Dict,
    bare_forces: Dict,
    use_forces: bool,
):
    """
    Build lists of (structure, energy_per_atom, forces) for CHGNet StructureData.
    Includes both cluster+H and bare cluster structures.
    """
    structures = []
    energies_per_atom = []
    forces_list = []

    for site in sites:
        cid = site["config_id"]
        n_cu = site["size"]
        n_total = n_cu + 1  # cluster + H

        # Cluster+H structure
        atoms_clus_h = site_to_cluster_h_atoms(site, prepared)
        if atoms_clus_h is None:
            continue

        e_clus_h = site["e_clus_h_eV"]
        structures.append(ase_to_pmg(atoms_clus_h))
        energies_per_atom.append(e_clus_h / n_total)

        if use_forces and site.get("forces_eVA") is not None:
            f_arr = np.array(site["forces_eVA"])
            if f_arr.shape == (n_total, 3):
                forces_list.append(f_arr)
            else:
                forces_list.append(np.zeros((n_total, 3)))
        else:
            forces_list.append(np.zeros((n_total, 3)))

        # Bare cluster structure
        if cid in bare_clusters:
            atoms_bare = bare_cluster_atoms_from_traj(bare_clusters, cid)
            if atoms_bare is not None:
                e_bare = site["e_cluster_eV"]
                structures.append(ase_to_pmg(atoms_bare))
                energies_per_atom.append(e_bare / n_cu)

                if use_forces and cid in bare_forces:
                    f_bare = np.array(bare_forces[cid])
                    if f_bare.shape == (n_cu, 3):
                        forces_list.append(f_bare)
                    else:
                        forces_list.append(np.zeros((n_cu, 3)))
                else:
                    forces_list.append(np.zeros((n_cu, 3)))

    return structures, energies_per_atom, forces_list


def finetune_chgnet(
    structures: List,
    energies_per_atom: List,
    forces_list: List,
    save_dir: Path,
    run_name: str,
    seed: int,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> tuple:
    """
    Fine-tune CHGNet on the given training data.
    Returns (ckpt_path, elapsed_s) or (None, elapsed_s) on failure.
    """
    import torch
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    from chgnet.data.dataset import StructureData, get_train_val_test_loader

    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CHGNet.load()
    trainer = Trainer(
        model=model,
        targets="ef",
        epochs=epochs,
        learning_rate=lr,
        use_device="cuda",
        torch_seed=seed,
        data_seed=seed,
    )

    n_structures = len(structures)
    # val_ratio: keep at least 1 for val, but don't exceed 20%
    val_ratio = max(1.0 / n_structures, 0.2)
    train_ratio = 1.0 - val_ratio
    batch_size = min(4, max(1, int(train_ratio * n_structures)))

    dataset = StructureData(
        structures=structures,
        energies=energies_per_atom,
        forces=forces_list,
    )

    if len(dataset) < 2:
        print(f"    [FAIL] {run_name}: dataset too small ({len(dataset)} structures)")
        return None

    loaders = get_train_val_test_loader(
        dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        return_test=False,
        num_workers=0,
    )
    train_loader, val_loader = loaders

    t0 = time.time()

    try:
        trainer.train(train_loader, val_loader, save_dir=str(save_dir))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    [FAIL] {run_name}: training exception: {e}")
        return None, elapsed

    elapsed = time.time() - t0

    # Find best energy checkpoint
    best_e_files = sorted(save_dir.glob("bestE_*.pth.tar"))
    if not best_e_files:
        best_e_files = sorted(save_dir.glob("*.pth.tar"))
    if not best_e_files:
        print(f"    [FAIL] {run_name}: no checkpoint saved")
        return None, elapsed

    best_ckpt = best_e_files[-1]
    print(f"    [OK] {run_name} — {elapsed:.0f}s → {best_ckpt.name}")
    return best_ckpt, elapsed


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_finetuned_chgnet(
    ckpt_path: Path,
    test_sites: List[Dict],
    prepared: Dict,
    bare_clusters: Dict,
    e_h2_chgnet: float,
) -> Dict:
    """
    Evaluate fine-tuned CHGNet on test sites.
    Computes ΔGH* = E(cluster+H) - E(cluster) - 0.5*E(H2) + ZPE_CORR
    using the fine-tuned model for all energy evaluations.
    """
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    ft_model = CHGNet.from_file(str(ckpt_path))
    calc = CHGNetCalculator(model=ft_model, use_device="cuda")

    predictions = []
    for site in test_sites:
        cid = site["config_id"]
        n_cu = site["size"]

        # Bare cluster
        atoms_bare = bare_cluster_atoms_from_traj(bare_clusters, cid)
        if atoms_bare is None:
            continue
        atoms_bare.calc = calc
        try:
            e_cluster = float(atoms_bare.get_potential_energy())
        except Exception as ex:
            print(f"      [skip] bare cluster {cid}: {ex}")
            continue

        # Cluster+H
        atoms_clus_h = site_to_cluster_h_atoms(site, prepared)
        if atoms_clus_h is None:
            continue
        atoms_clus_h.calc = calc
        try:
            e_clus_h = float(atoms_clus_h.get_potential_energy())
        except Exception as ex:
            print(f"      [skip] cluster+H {site['site_global_id']}: {ex}")
            continue

        dgh_ft = e_clus_h - e_cluster - 0.5 * e_h2_chgnet + ZPE_CORR
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
    ft_vals   = np.array([p["dgh_ft_eV"]  for p in predictions])

    return {
        "mae":       float(np.mean(np.abs(errors))),
        "rmse":      float(np.sqrt(np.mean(errors ** 2))),
        "pearson_r": float(np.corrcoef(gpaw_vals, ft_vals)[0, 1]) if len(predictions) > 1 else 0.0,
        "n":         len(predictions),
        "predictions": predictions,
    }


# ── Fig 8 Update ───────────────────────────────────────────────────────────────

def plot_learning_curves_updated(
    mace_lc: Dict,
    chgnet_lc: Dict,
    zero_shot: Dict,
    output_path: Path,
):
    """
    Updated Fig 8: Learning curves for both MACE-MP-0 and CHGNet fine-tuning.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Learning Curves: Fine-tuned MACE-MP-0 and CHGNet vs Foundation Models\n"
        r"Cu$_{10}$–Cu$_{50}$ Nanoclusters, H* Adsorption (ΔG$_H^*$)",
        fontsize=13, fontweight="bold",
    )

    mace_ft_color  = "#e74c3c"
    chg_ft_color   = "#27ae60"
    mace_zs_color  = "#3498db"
    chg_zs_color   = "#2ecc71"
    tn_color       = "#9b59b6"

    all_train_sizes = sorted(set(list(mace_lc.keys()) + list(chgnet_lc.keys())))

    for ax, metric in zip(axes, ["mae", "pearson_r"]):
        ylabel = "MAE (eV)" if metric == "mae" else "Pearson r"
        title  = "Accuracy (MAE)" if metric == "mae" else "Rank-order fidelity (Pearson r)"

        # MACE fine-tuned
        if mace_lc:
            ns    = sorted(mace_lc.keys())
            means = [mace_lc[n][f"{metric}_mean"] for n in ns]
            stds  = [mace_lc[n][f"{metric}_std"]  for n in ns]
            ax.plot(ns, means, "o-", color=mace_ft_color, lw=2, ms=7,
                    label="MACE-MP-0 fine-tuned", zorder=3)
            ax.fill_between(ns,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=mace_ft_color)

        # CHGNet fine-tuned
        if chgnet_lc:
            ns    = sorted(chgnet_lc.keys())
            means = [chgnet_lc[n][f"{metric}_mean"] for n in ns]
            stds  = [chgnet_lc[n][f"{metric}_std"]  for n in ns]
            ax.plot(ns, means, "s-", color=chg_ft_color, lw=2, ms=7,
                    label="CHGNet fine-tuned", zorder=3)
            ax.fill_between(ns,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=chg_ft_color)

        # Zero-shot baselines
        ax.axhline(zero_shot.get("MACE-MP-0", {}).get(metric, np.nan),
                   ls="--", color=mace_zs_color, lw=1.5, label="MACE-MP-0 zero-shot")
        ax.axhline(zero_shot.get("CHGNet", {}).get(metric, np.nan),
                   ls="-.", color=chg_zs_color,  lw=1.5, label="CHGNet zero-shot")
        ax.axhline(zero_shot.get("TensorNet", {}).get(metric, np.nan),
                   ls=":",  color=tn_color,       lw=1.5, label="TensorNet zero-shot")

        ax.set_xlabel("Training set size N (DFT points)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        if metric == "mae":
            ax.set_ylim(bottom=0)
        if all_train_sizes:
            ax.set_xticks(all_train_sizes)
        ax.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import random
    import torch

    VIZ_DIR.mkdir(exist_ok=True)
    CHGNET_FT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 13 — CHGNet Fine-tuning + Learning Curves")
    print("=" * 65)

    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU required.")
        sys.exit(1)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading GPAW data...")
    gpaw_sites = load_all_gpaw_sites()
    print(f"  Total: {len(gpaw_sites)} sites")
    if len(gpaw_sites) < 20:
        print("ERROR: Not enough GPAW data.")
        sys.exit(1)

    prepared      = load_prepared_sites()
    bare_clusters = load_bare_cluster_structures()
    bare_forces   = load_bare_cluster_forces()

    first_with_forces = next(
        (s for s in gpaw_sites if s.get("forces_eVA") is not None), None
    )
    USE_FORCES = (first_with_forces is not None) and bool(bare_forces)
    print(f"  Force training: {'ENABLED' if USE_FORCES else 'DISABLED (energy-only)'}")

    # ── 2. Same train/test split as Script 09 ───────────────────────────────
    print("\n[2] Building stratified train/test split (seed=42)...")
    rng = np.random.default_rng(seed=42)
    by_size = {}
    for site in gpaw_sites:
        by_size.setdefault(site["size"], []).append(site)

    test_sites = []
    train_pool = []
    for sz, sites in sorted(by_size.items()):
        n_test = max(1, round(len(sites) * TEST_FRAC))
        idx = rng.permutation(len(sites))
        test_sites.extend([sites[i] for i in idx[:n_test]])
        train_pool.extend([sites[i] for i in idx[n_test:]])

    print(f"  Train pool: {len(train_pool)} | Test: {len(test_sites)}")

    # ── 3. Load zero-shot baselines ─────────────────────────────────────────
    print("\n[3] Loading zero-shot baselines from benchmark_summary.json...")
    zero_shot = {}
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if summary_path.exists():
        bsum = json.load(open(summary_path))
        bm = bsum.get("by_model", {})
        for model_name, mdata in bm.items():
            if isinstance(mdata, dict) and "mae_eV" in mdata:
                zero_shot[model_name] = {
                    "mae":       mdata["mae_eV"],
                    "pearson_r": mdata.get("pearson_r", 0.0),
                }
        print(f"  Loaded: {list(zero_shot.keys())}")
    else:
        print("  [warn] benchmark_summary.json not found")

    # ── 4. Load CHGNet H2 reference energy ──────────────────────────────────
    print("\n[4] Loading CHGNet H2 reference energy...")
    h2_cache = RESULTS_DIR / "chgnet" / "h2_energy_chgnet.json"
    if h2_cache.exists():
        e_h2_chgnet = json.load(open(h2_cache))["e_h2_eV"]
        print(f"  E(H2) CHGNet = {e_h2_chgnet:.5f} eV (from cache)")
    else:
        print("  ERROR: CHGNet H2 cache not found. Run script 05 first.")
        sys.exit(1)

    # ── 5. Learning curve loop ───────────────────────────────────────────────
    print("\n[5] Running CHGNet fine-tuning learning curves...")
    chgnet_lc_results = {}

    for n_train in TRAIN_SIZES:
        if n_train > len(train_pool):
            print(f"\n  [skip] N={n_train}: not enough data ({len(train_pool)})")
            continue

        print(f"\n  ── N={n_train} training points ──")
        seed_results = []

        for seed in range(N_SEEDS):
            # Stratified sample from pool
            rng_seed = np.random.default_rng(seed=seed * 100 + n_train)
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

            run_name = f"chgnet_ft_N{n_train:03d}_seed{seed}"
            run_dir  = CHGNET_FT_DIR / run_name

            # Skip if already done
            eval_cache = run_dir / "eval_result.json"
            if eval_cache.exists():
                er = json.load(open(eval_cache))
                if er.get("mae") is not None:
                    print(f"    seed={seed}: [cached] MAE={er['mae']:.4f}  FT={er.get('ft_time_s','?'):.0f}s")
                    seed_results.append(er)
                    continue

            # Build dataset
            structures, energies_per_atom, forces_list = build_chgnet_dataset(
                sampled, prepared, bare_clusters, bare_forces, USE_FORCES
            )
            print(f"    seed={seed}: {len(structures)} structures → fine-tuning...")

            # Fine-tune
            ckpt, ft_elapsed = finetune_chgnet(
                structures=structures,
                energies_per_atom=energies_per_atom,
                forces_list=forces_list,
                save_dir=run_dir,
                run_name=run_name,
                seed=seed,
                epochs=EPOCHS,
                lr=LR,
            )
            if ckpt is None:
                print(f"    [WARN] Fine-tuning failed seed={seed}")
                continue

            # Evaluate
            print(f"    Evaluating on {len(test_sites)} test sites...")
            t_eval = time.time()
            eval_result = evaluate_finetuned_chgnet(
                ckpt_path=ckpt,
                test_sites=test_sites,
                prepared=prepared,
                bare_clusters=bare_clusters,
                e_h2_chgnet=e_h2_chgnet,
            )
            eval_result["ft_time_s"] = float(ft_elapsed)
            print(f"    MAE={eval_result['mae']:.4f} eV | "
                  f"RMSE={eval_result['rmse']:.4f} eV | "
                  f"r={eval_result['pearson_r']:.3f} (n={eval_result['n']}) "
                  f"[FT={ft_elapsed:.0f}s, eval={time.time()-t_eval:.0f}s]")
            seed_results.append(eval_result)
            json.dump(eval_result, open(run_dir / "eval_result.json", "w"), indent=2)

        if not seed_results:
            continue

        valid = [r for r in seed_results if r.get("mae") is not None]
        if not valid:
            continue

        ft_times = [r["ft_time_s"] for r in valid if r.get("ft_time_s") is not None]
        chgnet_lc_results[n_train] = {
            "mae_mean":        float(np.mean([r["mae"] for r in valid])),
            "mae_std":         float(np.std([r["mae"] for r in valid])),
            "rmse_mean":       float(np.mean([r["rmse"] for r in valid])),
            "rmse_std":        float(np.std([r["rmse"] for r in valid])),
            "pearson_r_mean":  float(np.mean([r["pearson_r"] for r in valid])),
            "pearson_r_std":   float(np.std([r["pearson_r"] for r in valid])),
            "n_seeds_ok":      len(valid),
            "ft_time_s_mean":  float(np.mean(ft_times)) if ft_times else None,
            "ft_time_s_std":   float(np.std(ft_times))  if ft_times else None,
        }
        v = chgnet_lc_results[n_train]
        print(f"\n  N={n_train} summary: MAE={v['mae_mean']:.3f}±{v['mae_std']:.3f} eV")

    # ── 6. Save results ──────────────────────────────────────────────────────
    print("\n[6] Saving CHGNet learning curve results...")
    chgnet_lc_path = RESULTS_DIR / "chgnet_lc.json"
    json.dump({
        "train_sizes": TRAIN_SIZES,
        "n_seeds":     N_SEEDS,
        "test_frac":   TEST_FRAC,
        "n_test":      len(test_sites),
        "zero_shot":   zero_shot,
        "learning_curve": chgnet_lc_results,
    }, open(chgnet_lc_path, "w"), indent=2)
    print(f"  Saved: {chgnet_lc_path}")

    # ── 7. Load MACE LC results and update Fig 8 ─────────────────────────────
    print("\n[7] Updating Fig 8 with CHGNet fine-tuning curve...")
    lc_path = RESULTS_DIR / "learning_curves.json"
    mace_lc = {}
    if lc_path.exists():
        mace_data = json.load(open(lc_path))
        raw_lc = mace_data.get("learning_curve", {})
        # Keys may be strings from JSON
        for k, v in raw_lc.items():
            mace_lc[int(k)] = v
        print(f"  MACE LC loaded: N={sorted(mace_lc.keys())}")
    else:
        print("  [warn] learning_curves.json not found — MACE curve will be absent")

    # Build zero-shot dict for plot (mae_mean key names)
    zs_plot = {}
    for model_name, metrics in zero_shot.items():
        zs_plot[model_name] = {
            "mae":       metrics.get("mae", np.nan),
            "pearson_r": metrics.get("pearson_r", np.nan),
        }

    # Convert chgnet_lc_results keys to int for plot
    chgnet_lc_plot = {int(k): v for k, v in chgnet_lc_results.items()}

    # MACE lc uses mean/std naming — expose as _mean/_std too
    mace_lc_plot = {}
    for n, v in mace_lc.items():
        mace_lc_plot[n] = {
            "mae_mean":       v.get("mae_mean", v.get("mae", np.nan)),
            "mae_std":        v.get("mae_std", 0.0),
            "pearson_r_mean": v.get("pearson_r_mean", v.get("pearson_r", np.nan)),
            "pearson_r_std":  v.get("pearson_r_std", 0.0),
        }

    fig8_path = VIZ_DIR / "fig8_learning_curves.png"
    plot_learning_curves_updated(mace_lc_plot, chgnet_lc_plot, zs_plot, fig8_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CHGNET LEARNING CURVE SUMMARY")
    print("=" * 65)
    print("\nZero-shot baselines:")
    for model, m in zero_shot.items():
        print(f"  {model:15s}: MAE={m.get('mae', 'n/a')} eV, r={m.get('pearson_r', 'n/a')}")

    if chgnet_lc_results:
        print("\nFine-tuned CHGNet MAE at each N:")
        for n, v in sorted(chgnet_lc_results.items()):
            print(f"  N={n:3d}: {v['mae_mean']:.3f}±{v['mae_std']:.3f} eV "
                  f"(r={v['pearson_r_mean']:.3f}±{v['pearson_r_std']:.3f})")

        best_n   = min(chgnet_lc_results, key=lambda n: chgnet_lc_results[n]["mae_mean"])
        best_mae = chgnet_lc_results[best_n]["mae_mean"]
        zs_chg   = zero_shot.get("CHGNet", {}).get("mae")
        if zs_chg:
            improvement = (zs_chg - best_mae) / zs_chg * 100
            print(f"\n  Best: N={best_n} → MAE={best_mae:.3f} eV "
                  f"({improvement:.0f}% improvement over zero-shot CHGNet)")

    print(f"\nFig 8: {fig8_path}")
    print("Done.")


if __name__ == "__main__":
    main()
