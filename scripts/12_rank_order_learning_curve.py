"""
Script 12 — Rank-Order Fidelity vs. Training Set Size (Fine-tuned MACE)
========================================================================
Answers the screening question: "As I add more DFT training points and fine-tune
MACE, how quickly does screening quality (rank-order fidelity) improve?"

Loads the 15 pre-computed eval_result.json files from script 09 (no re-inference
needed — per-site predictions are already stored there).

Computes per (N_train, seed):
  - Spearman ρ between fine-tuned MACE ΔGH* and GPAW ΔGH* on 29 test sites
  - top-10%, top-15%, top-20%, top-25% recall

Averages across 3 seeds with std.

Adds zero-shot baselines from rank_order_results.json for comparison.

Outputs:
  results/rank_order_lc.json      ← full results table
  viz/fig10_rank_order_lc.png     ← Spearman ρ + recall vs N panels

Run after script 09 completes (models + eval_result.json already present):
    python3 scripts/12_rank_order_learning_curve.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_DIR / "results"
FINETUNE_DIR = RESULTS_DIR / "finetune"
VIZ_DIR      = PROJECT_DIR / "viz"

TRAIN_SIZES = [10, 20, 40, 60, 80]
N_SEEDS     = 3

RECALL_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]  # top-K% of test set


def compute_recall(gpaw_vals, pred_vals, frac):
    """Fraction of GPAW top-K% sites found in predicted top-K% sites."""
    n = len(gpaw_vals)
    k = max(1, round(n * frac))
    gpaw_top = set(np.argsort(gpaw_vals)[:k])
    pred_top = set(np.argsort(pred_vals)[:k])
    return len(gpaw_top & pred_top) / k


def load_seed_predictions(n_train: int, seed: int):
    """Load per-site predictions from eval_result.json for one (N, seed) run."""
    name = f"mace_ft_N{n_train:03d}_seed{seed}"
    path = FINETUNE_DIR / name / "eval_result.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    return data.get("predictions", [])


def main():
    print("=" * 65)
    print("Script 12 — Rank-Order Fidelity vs. N (Fine-tuned MACE)")
    print("=" * 65)

    # ── 1. Load per-seed predictions and compute rank metrics ─────────────
    print("\n[1] Loading eval_result.json files...")
    lc_results = {}

    for n_train in TRAIN_SIZES:
        seed_metrics = []
        for seed in range(N_SEEDS):
            preds = load_seed_predictions(n_train, seed)
            if preds is None:
                print(f"  [skip] N={n_train} seed={seed}: no eval_result.json")
                continue
            if len(preds) == 0:
                print(f"  [skip] N={n_train} seed={seed}: empty predictions")
                continue

            gpaw_vals = np.array([p["dgh_gpaw_eV"] for p in preds])
            ft_vals   = np.array([p["dgh_ft_eV"]   for p in preds])

            spearman_r, spearman_p = stats.spearmanr(gpaw_vals, ft_vals)
            mae  = float(np.mean(np.abs(ft_vals - gpaw_vals)))
            rmse = float(np.sqrt(np.mean((ft_vals - gpaw_vals) ** 2)))

            recalls = {
                f"top{int(f*100)}pct_recall": compute_recall(gpaw_vals, ft_vals, f)
                for f in RECALL_THRESHOLDS
            }

            seed_metrics.append({
                "seed": seed,
                "n_preds": len(preds),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "mae_eV": mae,
                "rmse_eV": rmse,
                **recalls,
            })
            print(f"  N={n_train:3d} seed={seed}: ρ={spearman_r:.3f}  "
                  f"top10%={recalls['top10pct_recall']:.2f}  "
                  f"top20%={recalls['top20pct_recall']:.2f}  "
                  f"MAE={mae:.3f} eV  (n={len(preds)})")

        if not seed_metrics:
            continue

        # Aggregate across seeds
        def agg(key):
            vals = [m[key] for m in seed_metrics]
            return float(np.mean(vals)), float(np.std(vals))

        lc_results[n_train] = {
            "n_seeds":          len(seed_metrics),
            "spearman_r_mean":  agg("spearman_r")[0],
            "spearman_r_std":   agg("spearman_r")[1],
            "mae_mean":         agg("mae_eV")[0],
            "mae_std":          agg("mae_eV")[1],
            **{f"top{int(f*100)}pct_recall_mean": agg(f"top{int(f*100)}pct_recall")[0]
               for f in RECALL_THRESHOLDS},
            **{f"top{int(f*100)}pct_recall_std":  agg(f"top{int(f*100)}pct_recall")[1]
               for f in RECALL_THRESHOLDS},
            "per_seed": seed_metrics,
        }

    # ── 2. Load zero-shot baselines ───────────────────────────────────────
    print("\n[2] Loading zero-shot baselines...")
    zs = {}
    zs_path = RESULTS_DIR / "rank_order_results.json"
    if zs_path.exists():
        raw = json.load(open(zs_path))
        for model, metrics in raw["overall"].items():
            zs[model] = {
                "spearman_r":     metrics["spearman_r"],
                "top10pct_recall": metrics.get("top10pct_recall", 0.0),
                "top15pct_recall": metrics.get("top15pct_recall", 0.0),
                "top20pct_recall": metrics.get("top20pct_recall", 0.0),
                "top25pct_recall": metrics.get("top25pct_recall", 0.0),
            }
        print(f"  Loaded: {list(zs.keys())}")

    # ── 3. Save results ───────────────────────────────────────────────────
    print("\n[3] Saving results...")
    out_path = RESULTS_DIR / "rank_order_lc.json"
    json.dump({
        "train_sizes": TRAIN_SIZES,
        "zero_shot":   zs,
        "learning_curve": {str(n): v for n, v in lc_results.items()},
    }, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")

    # ── 4. Print summary table ────────────────────────────────────────────
    print("\n[4] Summary — Rank-Order Fidelity vs. N (fine-tuned MACE vs zero-shot)")
    print(f"\n{'N':>6}  {'Spearman ρ':>12}  {'top-10%':>9}  {'top-20%':>9}")
    print("-" * 45)
    # Zero-shot baselines
    for model, m in zs.items():
        print(f"  {'ZS-'+model[:4]:>6}  {m['spearman_r']:>12.3f}  "
              f"{m['top10pct_recall']:>9.2f}  {m['top20pct_recall']:>9.2f}")
    print("-" * 45)
    for n in TRAIN_SIZES:
        if n not in lc_results:
            continue
        r = lc_results[n]
        print(f"  {n:>6}  "
              f"{r['spearman_r_mean']:>10.3f}±{r['spearman_r_std']:.2f}  "
              f"{r['top10pct_recall_mean']:>9.2f}  "
              f"{r['top20pct_recall_mean']:>9.2f}")

    # ── 5. Plot fig10 ─────────────────────────────────────────────────────
    print("\n[5] Generating fig10_rank_order_lc.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"Rank-Order Fidelity vs. Fine-Tuning Dataset Size (MACE-MP-0)"
        "\n" + r"Cu$_{10}$–Cu$_{50}$ Nanoclusters, H* Adsorption (ΔG$_H^*$)",
        fontsize=13, fontweight="bold",
    )

    # Colors
    ft_color   = "#e74c3c"   # fine-tuned MACE
    mace_zs    = "#3498db"   # MACE zero-shot
    chg_zs     = "#2ecc71"   # CHGNet zero-shot
    tn_zs      = "#9b59b6"   # TensorNet zero-shot

    model_colors = {"MACE-MP-0": mace_zs, "CHGNet": chg_zs, "TensorNet": tn_zs}
    model_styles = {"MACE-MP-0": "--", "CHGNet": "-.", "TensorNet": ":"}
    model_labels = {"MACE-MP-0": "MACE-MP-0 (zero-shot)",
                    "CHGNet": "CHGNet (zero-shot)",
                    "TensorNet": "TensorNet (zero-shot)"}

    ns = sorted(lc_results.keys())

    for ax, (metric_key, recall_key, ylabel, title) in zip(
        axes,
        [
            ("spearman_r_mean",     "spearman_r_std",
             "Spearman ρ", "Rank Correlation (Spearman ρ)"),
            ("top20pct_recall_mean", "top20pct_recall_std",
             "Top-20% Recall", "Screening Recall (Top-20%)"),
        ]
    ):
        means = [lc_results[n][metric_key]  for n in ns]
        stds  = [lc_results[n][recall_key]  for n in ns]

        ax.plot(ns, means, "o-", color=ft_color, lw=2.5, ms=8,
                label="MACE-MP-0 (fine-tuned)", zorder=4)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=ft_color)

        # Zero-shot baselines
        for model, m in zs.items():
            zs_val = m.get(metric_key.replace("_mean", ""), None)
            if zs_val is None:
                # map recall key
                if "top20" in metric_key:
                    zs_val = m.get("top20pct_recall", None)
                else:
                    zs_val = m.get("spearman_r", None)
            if zs_val is None:
                continue
            ax.axhline(zs_val,
                       ls=model_styles.get(model, "--"),
                       color=model_colors.get(model, "gray"),
                       lw=1.8,
                       label=model_labels.get(model, model),
                       zorder=2)

        ax.set_xlabel("Fine-tuning dataset size N (DFT calculations)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        if "spearman" in metric_key:
            ax.set_ylim(0, 1)

    plt.tight_layout()
    out_fig = VIZ_DIR / "fig10_rank_order_lc.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_fig}")

    print("\nDone — fig10 and rank_order_lc.json complete.")


if __name__ == "__main__":
    main()
