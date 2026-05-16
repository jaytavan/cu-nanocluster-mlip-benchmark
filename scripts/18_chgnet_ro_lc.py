"""
Script 18 — CHGNet Fine-tuning Rank-Order Learning Curve (ALL 364 sites)
=========================================================================
Uses all 364 GPAW reference sites (Cu10–Cu50, 3 geometries per size).

Goal: How quickly does CHGNet rank-order fidelity (Spearman ρ, top-k% recall)
improve as we add fine-tuning data?

Approach:
- Same stratified train/test split as scripts 09/09b (seed=42, 20% test)
- Fine-tune CHGNet at N = [10, 20, 40, 60, 80] × 3 seeds (fast: 9–91s each)
- Store per-site ΔGH* predictions after each run
- Compute Spearman ρ, top-10%, top-20% recall vs GPAW reference
- Saves results to:
    results/chgnet_ro_lc.json          ← aggregated learning curve
    results/chgnet_ro_lc_predictions/  ← per-run per-site predictions

IMPORTANT: CHGNet needs pbc=True — structures are given a 30Å vacuum cell.

Run:
    nohup /home/jay/miniconda3/envs/catalyst/bin/python3 -u scripts/18_chgnet_ro_lc.py \
        > results/chgnet_ro_lc_run.log 2>&1 &

Prerequisites:
    results/gpaw/{Cu10..Cu50}/results_Cu{N}_{00,01,02}.json
    results/gpaw/sites_prepared.json
    results/clusters_relaxed.traj
    results/gpaw/Cu{N}/cluster_energy_Cu{N}_{00,01,02}_d3.json
    results/chgnet/h2_energy_chgnet.json
"""

import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).parent
PROJECT_DIR      = SCRIPT_DIR.parent
RESULTS_DIR      = PROJECT_DIR / "results"
GPAW_DIR         = RESULTS_DIR / "gpaw"
VIZ_DIR          = PROJECT_DIR / "viz"
CHGNET_FT_DIR    = RESULTS_DIR / "chgnet_ft_ro"
PRED_DIR         = RESULTS_DIR / "chgnet_ro_lc_predictions"

ZPE_CORR         = 0.24   # eV
TRAIN_SIZES      = [10, 20, 40, 60, 80]
N_SEEDS          = 3
TEST_FRAC        = 0.20
EPOCHS           = 50
LR               = 1e-3
CLUSTER_SUFFIXES = ["00", "01", "02"]
RECALL_THRESHOLDS = [0.10, 0.20]   # top-10%, top-20%


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_gpaw_sites() -> List[Dict]:
    """Load all GPAW sites across all 3 geometries per size."""
    all_sites = []
    for sz in [10, 20, 30, 40, 50]:
        sz_count = 0
        for suffix in CLUSTER_SUFFIXES:
            result_file = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_{suffix}.json"
            if not result_file.exists():
                continue
            data = json.load(open(result_file))
            ok_sites = [s for s in data["sites"] if s["status"] == "ok"]
            sz_count += len(ok_sites)
            all_sites.extend(ok_sites)
        print(f"  Cu{sz}: {sz_count} sites (3 geometries)")
    return all_sites


def load_prepared_sites() -> Dict[str, Dict]:
    prep = json.load(open(GPAW_DIR / "sites_prepared.json"))
    return {s["site_global_id"]: s for s in prep}


def load_bare_cluster_structures() -> Dict[str, "ase.Atoms"]:
    from ase.io import read
    traj = read(str(RESULTS_DIR / "clusters_relaxed.traj"), index=":")
    clusters = {}
    for atoms in traj:
        cid = atoms.info.get("config_id")
        if cid is None:
            cid = f"Cu{len(atoms)}_00"
        if cid not in clusters:
            clusters[cid] = atoms
    print(f"  Bare cluster structures: {len(clusters)} entries")
    return clusters


def load_bare_cluster_forces() -> Dict[str, List]:
    forces_map = {}
    for sz in [10, 20, 30, 40, 50]:
        for suffix in CLUSTER_SUFFIXES:
            cid        = f"Cu{sz}_{suffix}"
            cache_file = GPAW_DIR / f"Cu{sz}" / f"cluster_energy_{cid}_d3.json"
            if cache_file.exists():
                data = json.load(open(cache_file))
                if data.get("forces_eVA") is not None:
                    forces_map[cid] = data["forces_eVA"]
    print(f"  Bare cluster forces: {len(forces_map)} configs loaded")
    return forces_map


# ── ASE helpers ───────────────────────────────────────────────────────────────

def site_to_cluster_h_atoms(gpaw_site: Dict, prepared: Dict):
    """Reconstruct cluster+H ASE Atoms (with 30Å vacuum for pbc=True)."""
    from ase import Atoms
    sid = gpaw_site["site_global_id"]
    if sid not in prepared:
        return None
    prep        = prepared[sid]
    n_cu        = gpaw_site["size"]
    cluster_pos = np.array(prep["cluster_pos"])
    h_pos       = np.array(prep["h_pos"])
    cell        = np.array(prep["cell"])
    positions   = np.vstack([cluster_pos, h_pos.reshape(1, 3)])
    atoms = Atoms(symbols=["Cu"] * n_cu + ["H"], positions=positions, cell=cell)
    atoms.center(vacuum=15.0)   # 30 Å total vacuum (15 Å each side)
    atoms.set_pbc(True)
    return atoms


def bare_cluster_atoms(cluster) -> "ase.Atoms":
    """Bare cluster with 30Å vacuum cell and pbc=True (required by CHGNet)."""
    atoms = cluster.copy()
    atoms.center(vacuum=15.0)
    atoms.set_pbc(True)
    return atoms


def ase_to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


# ── CHGNet dataset builder ────────────────────────────────────────────────────

def build_chgnet_dataset(
    sites: List[Dict],
    prepared: Dict,
    bare_clusters: Dict,
    bare_forces: Dict,
    use_forces: bool,
):
    """Build (structures, energies_per_atom, forces) lists for CHGNet StructureData."""
    structures         = []
    energies_per_atom  = []
    forces_list        = []

    for site in sites:
        cid    = site["config_id"]
        n_cu   = site["size"]
        n_tot  = n_cu + 1  # cluster + H

        atoms_clus_h = site_to_cluster_h_atoms(site, prepared)
        if atoms_clus_h is None:
            continue

        e_clus_h = site["e_clus_h_eV"]
        structures.append(ase_to_pmg(atoms_clus_h))
        energies_per_atom.append(e_clus_h / n_tot)

        if use_forces and site.get("forces_eVA") is not None:
            f_arr = np.array(site["forces_eVA"])
            forces_list.append(f_arr if f_arr.shape == (n_tot, 3)
                               else np.zeros((n_tot, 3)))
        else:
            forces_list.append(np.zeros((n_tot, 3)))

        # Bare cluster
        cluster = bare_clusters.get(cid)
        if cluster is not None:
            atoms_bare = bare_cluster_atoms(cluster)
            e_bare     = site["e_cluster_eV"]
            structures.append(ase_to_pmg(atoms_bare))
            energies_per_atom.append(e_bare / n_cu)

            if use_forces and cid in bare_forces:
                f_b = np.array(bare_forces[cid])
                forces_list.append(f_b if f_b.shape == (n_cu, 3)
                                   else np.zeros((n_cu, 3)))
            else:
                forces_list.append(np.zeros((n_cu, 3)))

    return structures, energies_per_atom, forces_list


# ── CHGNet fine-tuner ─────────────────────────────────────────────────────────

def finetune_chgnet(
    structures: List,
    energies_per_atom: List,
    forces_list: List,
    save_dir: Path,
    run_name: str,
    seed: int,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    """
    Fine-tune CHGNet on given training data.
    Returns (ckpt_path, elapsed_s) or (None, elapsed_s) on failure.
    """
    import torch
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer
    from chgnet.data.dataset import StructureData, get_train_val_test_loader

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model   = CHGNet.load()
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
    val_ratio    = max(1.0 / n_structures, 0.2)
    train_ratio  = 1.0 - val_ratio
    batch_size   = min(4, max(1, int(train_ratio * n_structures)))

    dataset = StructureData(
        structures=structures,
        energies=energies_per_atom,
        forces=forces_list,
    )

    if len(dataset) < 2:
        print(f"    [FAIL] {run_name}: dataset too small ({len(dataset)})")
        return None, 0.0

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
        print(f"    [FAIL] {run_name}: {e}")
        return None, elapsed

    elapsed = time.time() - t0

    best_files = sorted(save_dir.glob("bestE_*.pth.tar"))
    if not best_files:
        best_files = sorted(save_dir.glob("*.pth.tar"))
    if not best_files:
        print(f"    [FAIL] {run_name}: no checkpoint saved")
        return None, elapsed

    best_ckpt = best_files[-1]
    print(f"    [OK] {run_name} — {elapsed:.0f}s → {best_ckpt.name}")
    return best_ckpt, elapsed


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_chgnet(
    ckpt_path: Path,
    test_sites: List[Dict],
    prepared: Dict,
    bare_clusters: Dict,
    e_h2_chgnet: float,
) -> Dict:
    """
    Evaluate fine-tuned CHGNet on test sites.
    Returns dict with predictions list and rank metrics.
    """
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    ft_model = CHGNet.from_file(str(ckpt_path))
    calc     = CHGNetCalculator(model=ft_model, use_device="cuda")

    predictions = []
    for site in test_sites:
        cid  = site["config_id"]
        n_cu = site["size"]

        cluster = bare_clusters.get(cid)
        if cluster is None:
            cid_fallback = f"Cu{n_cu}_00"
            cluster = bare_clusters.get(cid_fallback)
        if cluster is None:
            continue

        atoms_bare = bare_cluster_atoms(cluster)
        atoms_bare.calc = calc
        try:
            e_cluster = float(atoms_bare.get_potential_energy())
        except Exception as ex:
            print(f"      [skip] bare {cid}: {ex}")
            continue

        atoms_clus_h = site_to_cluster_h_atoms(site, prepared)
        if atoms_clus_h is None:
            continue
        atoms_clus_h.calc = calc
        try:
            e_clus_h = float(atoms_clus_h.get_potential_energy())
        except Exception as ex:
            print(f"      [skip] cluster+H {site['site_global_id']}: {ex}")
            continue

        dgh_ft   = e_clus_h - e_cluster - 0.5 * e_h2_chgnet + ZPE_CORR
        dgh_gpaw = site["dgh_gpaw_eV"]

        predictions.append({
            "site_global_id": site["site_global_id"],
            "size":           n_cu,
            "dgh_pred_eV":    float(dgh_ft),
            "dgh_ft_eV":      float(dgh_ft),
            "dgh_gpaw_eV":    float(dgh_gpaw),
            "error_eV":       float(dgh_ft - dgh_gpaw),
        })

    if not predictions:
        return {"mae": None, "rmse": None, "pearson_r": None,
                "spearman_r": None, "n": 0, "predictions": []}

    gpaw_vals = np.array([p["dgh_gpaw_eV"] for p in predictions])
    ft_vals   = np.array([p["dgh_ft_eV"]   for p in predictions])
    errors    = ft_vals - gpaw_vals

    mae        = float(np.mean(np.abs(errors)))
    rmse       = float(np.sqrt(np.mean(errors ** 2)))
    pearson_r  = float(np.corrcoef(gpaw_vals, ft_vals)[0, 1]) if len(predictions) > 1 else 0.0
    spearman_r = float(scipy_stats.spearmanr(gpaw_vals, ft_vals).correlation) if len(predictions) > 1 else 0.0

    recalls = {}
    for frac in RECALL_THRESHOLDS:
        k         = max(1, round(len(predictions) * frac))
        gpaw_top  = set(np.argsort(gpaw_vals)[:k])
        pred_top  = set(np.argsort(ft_vals)[:k])
        recalls[f"top{int(frac*100)}pct_recall"] = float(len(gpaw_top & pred_top) / k)

    return {
        "mae":        mae,
        "rmse":       rmse,
        "pearson_r":  pearson_r,
        "spearman_r": spearman_r,
        "n":          len(predictions),
        "predictions": predictions,
        **recalls,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import torch

    VIZ_DIR.mkdir(exist_ok=True)
    CHGNET_FT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 18 — CHGNet Fine-tuning Rank-Order LC (364 sites)")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU required.")
        sys.exit(1)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading GPAW data (all 3 geometries per size)...")
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

    # ── 2. Same train/test split as script 09/09b (seed=42) ───────────────
    print("\n[2] Building stratified train/test split (seed=42)...")
    rng = np.random.default_rng(seed=42)

    by_size = {}
    for site in gpaw_sites:
        by_size.setdefault(site["size"], []).append(site)

    test_sites = []
    train_pool = []
    for sz, sites in sorted(by_size.items()):
        n_test = max(1, round(len(sites) * TEST_FRAC))
        idx    = rng.permutation(len(sites))
        test_sites.extend([sites[i] for i in idx[:n_test]])
        train_pool.extend([sites[i] for i in idx[n_test:]])

    print(f"  Train pool: {len(train_pool)} | Test: {len(test_sites)}")

    # ── 3. Load CHGNet H2 reference energy ────────────────────────────────
    print("\n[3] Loading CHGNet H2 reference energy...")
    h2_cache_path = RESULTS_DIR / "chgnet" / "h2_energy_chgnet.json"
    if not h2_cache_path.exists():
        print("ERROR: CHGNet H2 cache not found. Run script 05 first.")
        sys.exit(1)
    e_h2_chgnet = json.load(open(h2_cache_path))["e_h2_eV"]
    print(f"  E(H2) CHGNet = {e_h2_chgnet:.5f} eV")

    # ── 4. Fine-tuning learning curve loop ────────────────────────────────
    print("\n[4] Running CHGNet fine-tuning learning curves...")
    lc_results = {}

    for n_train in TRAIN_SIZES:
        if n_train > len(train_pool):
            print(f"\n  [skip] N={n_train}: not enough data ({len(train_pool)})")
            continue

        print(f"\n  ── N={n_train} training points ──")
        seed_results = []

        for seed in range(N_SEEDS):
            run_name   = f"chgnet_ro_N{n_train:03d}_seed{seed}"
            run_dir    = CHGNET_FT_DIR / run_name
            eval_cache = run_dir / "eval_result.json"
            pred_cache = PRED_DIR / f"N{n_train:03d}_seed{seed}.json"

            # Resume from cache
            if eval_cache.exists():
                er = json.load(open(eval_cache))
                if er.get("mae") is not None:
                    print(f"    seed={seed}: [cached] "
                          f"MAE={er['mae']:.4f}  ρ={er.get('spearman_r','?'):.3f}  "
                          f"top10%={er.get('top10pct_recall','?'):.2f}")
                    seed_results.append(er)
                    continue

            # Stratified sample from pool
            rng_seed  = np.random.default_rng(seed=seed * 100 + n_train)
            sampled   = []
            remaining = n_train
            for sz in sorted(by_size.keys()):
                pool_sz   = [s for s in train_pool if s["size"] == sz]
                n_from_sz = max(1, round(n_train * len(pool_sz) / len(train_pool)))
                n_from_sz = min(n_from_sz, len(pool_sz), remaining)
                if n_from_sz == 0:
                    continue
                idx = rng_seed.choice(len(pool_sz), size=n_from_sz, replace=False)
                sampled.extend([pool_sz[i] for i in idx])
                remaining -= n_from_sz

            structures, energies_per_atom, forces_list = build_chgnet_dataset(
                sampled, prepared, bare_clusters, bare_forces, USE_FORCES
            )
            print(f"    seed={seed}: {len(structures)} structures → fine-tuning...")

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

            print(f"    Evaluating on {len(test_sites)} test sites...")
            t_eval = time.time()
            eval_result = evaluate_chgnet(
                ckpt_path=ckpt,
                test_sites=test_sites,
                prepared=prepared,
                bare_clusters=bare_clusters,
                e_h2_chgnet=e_h2_chgnet,
            )
            eval_result["ft_time_s"] = float(ft_elapsed)
            eval_result["n_train"]   = n_train
            eval_result["seed"]      = seed

            print(f"    MAE={eval_result['mae']:.4f} eV | "
                  f"ρ={eval_result['spearman_r']:.3f} | "
                  f"top10%={eval_result.get('top10pct_recall',0):.2f} | "
                  f"top20%={eval_result.get('top20pct_recall',0):.2f} | "
                  f"[FT={ft_elapsed:.0f}s, eval={time.time()-t_eval:.0f}s]")

            seed_results.append(eval_result)

            # Save eval_result.json (with predictions)
            json.dump(eval_result, open(eval_cache, "w"), indent=2)

            # Save per-run predictions to dedicated predictions dir
            per_site_preds = [
                {
                    "site_global_id": p["site_global_id"],
                    "dgh_pred_eV":    p["dgh_pred_eV"],
                    "dgh_gpaw_eV":    p["dgh_gpaw_eV"],
                }
                for p in eval_result.get("predictions", [])
            ]
            json.dump(per_site_preds, open(pred_cache, "w"), indent=2)

        if not seed_results:
            continue

        valid = [r for r in seed_results if r.get("mae") is not None]
        if not valid:
            continue

        def agg(key):
            vals = [r[key] for r in valid if r.get(key) is not None]
            return float(np.mean(vals)), float(np.std(vals))

        ft_times = [r["ft_time_s"] for r in valid if r.get("ft_time_s") is not None]

        lc_results[n_train] = {
            "n_seeds_ok":          len(valid),
            "mae_mean":            agg("mae")[0],
            "mae_std":             agg("mae")[1],
            "rmse_mean":           agg("rmse")[0],
            "rmse_std":            agg("rmse")[1],
            "pearson_r_mean":      agg("pearson_r")[0],
            "pearson_r_std":       agg("pearson_r")[1],
            "spearman_r_mean":     agg("spearman_r")[0],
            "spearman_r_std":      agg("spearman_r")[1],
            "top10pct_recall_mean": agg("top10pct_recall")[0],
            "top10pct_recall_std":  agg("top10pct_recall")[1],
            "top20pct_recall_mean": agg("top20pct_recall")[0],
            "top20pct_recall_std":  agg("top20pct_recall")[1],
            "ft_time_s_mean":      float(np.mean(ft_times)) if ft_times else None,
        }
        v = lc_results[n_train]
        print(f"\n  N={n_train} summary: "
              f"MAE={v['mae_mean']:.3f}±{v['mae_std']:.3f} eV  "
              f"ρ={v['spearman_r_mean']:.3f}±{v['spearman_r_std']:.3f}  "
              f"top10%={v['top10pct_recall_mean']:.2f}")

    # ── 5. Load zero-shot baselines ────────────────────────────────────────
    print("\n[5] Loading zero-shot baselines from rank_order_results.json...")
    zs = {}
    zs_path = RESULTS_DIR / "rank_order_results.json"
    if zs_path.exists():
        raw = json.load(open(zs_path))
        for model, metrics in raw.get("overall", {}).items():
            zs[model] = {
                "spearman_r":      metrics.get("spearman_r", 0.0),
                "top10pct_recall": metrics.get("top10pct_recall", 0.0),
                "top20pct_recall": metrics.get("top20pct_recall", 0.0),
            }
        print(f"  Loaded: {list(zs.keys())}")
    else:
        print("  [warn] rank_order_results.json not found — baselines absent")

    # ── 6. Save results ────────────────────────────────────────────────────
    print("\n[6] Saving CHGNet rank-order LC results...")
    out_path = RESULTS_DIR / "chgnet_ro_lc.json"
    json.dump({
        "train_sizes":    TRAIN_SIZES,
        "n_seeds":        N_SEEDS,
        "test_frac":      TEST_FRAC,
        "n_test":         len(test_sites),
        "n_total_sites":  len(gpaw_sites),
        "zero_shot":      zs,
        "learning_curve": {str(n): v for n, v in lc_results.items()},
    }, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")

    # ── 7. Plot ────────────────────────────────────────────────────────────
    print("\n[7] Generating viz/fig18_chgnet_ro_lc.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "CHGNet Fine-tuning: Rank-Order Fidelity vs. Training Set Size\n"
        r"Cu$_{10}$–Cu$_{50}$ Nanoclusters (364 sites), H* Adsorption (ΔG$_H^*$)",
        fontsize=13, fontweight="bold",
    )

    ft_color  = "#27ae60"   # CHGNet fine-tuned
    zs_colors = {"MACE-MP-0": "#3498db", "CHGNet": "#2ecc71", "TensorNet": "#9b59b6"}
    zs_styles = {"MACE-MP-0": "--", "CHGNet": "-.", "TensorNet": ":"}

    ns = sorted(lc_results.keys())

    for ax, (metric_key, std_key, ylabel, title) in zip(axes, [
        ("spearman_r_mean",      "spearman_r_std",
         "Spearman ρ",           "Rank Correlation (Spearman ρ)"),
        ("top20pct_recall_mean", "top20pct_recall_std",
         "Top-20% Recall",       "Screening Recall (Top-20%)"),
    ]):
        means = [lc_results[n][metric_key] for n in ns]
        stds  = [lc_results[n][std_key]    for n in ns]

        ax.plot(ns, means, "s-", color=ft_color, lw=2.5, ms=8,
                label="CHGNet (fine-tuned)", zorder=4)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=ft_color)

        for model, m in zs.items():
            val = m.get(metric_key.replace("_mean", ""), None)
            if val is None and "top20" in metric_key:
                val = m.get("top20pct_recall", None)
            elif val is None and "spearman" in metric_key:
                val = m.get("spearman_r", None)
            if val is None:
                continue
            ax.axhline(val,
                       ls=zs_styles.get(model, "--"),
                       color=zs_colors.get(model, "gray"),
                       lw=1.8, label=f"{model} (zero-shot)", zorder=2)

        ax.set_xlabel("Fine-tuning dataset size N (DFT calculations)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        if "spearman" in metric_key:
            ax.set_ylim(0, 1)
        if ns:
            ax.set_xticks(ns)
        ax.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    out_fig = VIZ_DIR / "fig18_chgnet_ro_lc.png"
    fig.savefig(str(out_fig), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_fig}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CHGNET RANK-ORDER LC SUMMARY (364 sites)")
    print("=" * 65)
    print(f"\n{'N':>6}  {'MAE (eV)':>12}  {'Spearman ρ':>12}  "
          f"{'top-10%':>9}  {'top-20%':>9}  {'FT (s)':>8}")
    print("-" * 60)
    for n in TRAIN_SIZES:
        if n not in lc_results:
            continue
        r = lc_results[n]
        ft_t = r.get("ft_time_s_mean")
        print(f"  {n:>4}  "
              f"{r['mae_mean']:>8.3f}±{r['mae_std']:.3f}  "
              f"{r['spearman_r_mean']:>8.3f}±{r['spearman_r_std']:.2f}  "
              f"{r['top10pct_recall_mean']:>9.2f}  "
              f"{r['top20pct_recall_mean']:>9.2f}  "
              f"{ft_t:>8.0f}" if ft_t else "")
    print(f"\nResults: {out_path}")
    print(f"Fig 18:  {out_fig}")
    print(f"Per-run predictions: {PRED_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
