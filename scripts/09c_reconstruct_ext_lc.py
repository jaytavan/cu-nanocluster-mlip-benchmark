"""
09c_reconstruct_ext_lc.py

Reconstruct learning_curves_ext.json and rank_order_lc_ext.json
from the eval_result.json prediction arrays in finetune_ext/.

Covers N=10,20,40,60,80 (3 seeds each) and N=100 (2 seeds).
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent
EXT_DIR = PROJECT / "results" / "finetune_ext"
OUT_DIR = PROJECT / "results"

TRAIN_SIZES = [10, 20, 40, 60, 80, 100]
N_SEEDS = 3


def recall_at_k(true_dg, pred_dg, k_frac):
    n = len(true_dg)
    k = max(1, int(np.ceil(n * k_frac)))
    true_top = set(np.argsort(true_dg)[:k])
    pred_top = set(np.argsort(pred_dg)[:k])
    return len(true_top & pred_top) / k


def process_seed(N, seed):
    d = EXT_DIR / f"mace_ft_N{N:03d}_seed{seed}" / "eval_result.json"
    if not d.exists():
        return None
    data = json.load(open(d))
    preds = data["predictions"]

    true_dg = np.array([p["dgh_gpaw_eV"] for p in preds])
    pred_dg = np.array([p["dgh_pred_eV"] for p in preds])
    err = pred_dg - true_dg

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    pearson_r = float(np.corrcoef(true_dg, pred_dg)[0, 1])
    sp_r = float(spearmanr(true_dg, pred_dg).statistic)

    r10 = recall_at_k(true_dg, pred_dg, 0.10)
    r15 = recall_at_k(true_dg, pred_dg, 0.15)
    r20 = recall_at_k(true_dg, pred_dg, 0.20)
    r25 = recall_at_k(true_dg, pred_dg, 0.25)

    return {
        "seed": seed,
        "n_preds": len(preds),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_r": sp_r,
        "recall_10": r10,
        "recall_15": r15,
        "recall_20": r20,
        "recall_25": r25,
    }


def agg(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)), float(np.std(v)), len(v)


def main():
    lc_entries = {}
    ro_entries = {}

    for N in TRAIN_SIZES:
        seed_results = []
        for seed in range(N_SEEDS):
            r = process_seed(N, seed)
            if r is not None:
                seed_results.append(r)
                print(f"  N={N:3d} seed{seed}: MAE={r['mae']:.3f}  spearman={r['spearman_r']:.3f}  recall10={r['recall_10']:.3f}")
            else:
                print(f"  N={N:3d} seed{seed}: MISSING")

        if not seed_results:
            continue

        key = str(N)

        # learning curves entry (MAE/RMSE/Pearson)
        mae_m, mae_s, n_ok = agg([r["mae"] for r in seed_results])
        rmse_m, rmse_s, _ = agg([r["rmse"] for r in seed_results])
        pr_m, pr_s, _ = agg([r["pearson_r"] for r in seed_results])
        lc_entries[key] = {
            "mae_mean": mae_m, "mae_std": mae_s,
            "rmse_mean": rmse_m, "rmse_std": rmse_s,
            "pearson_r_mean": pr_m, "pearson_r_std": pr_s,
            "n_seeds_ok": n_ok,
        }

        # rank-order entry (Spearman + recall)
        sp_m, sp_s, _ = agg([r["spearman_r"] for r in seed_results])
        r10_m, r10_s, _ = agg([r["recall_10"] for r in seed_results])
        r15_m, r15_s, _ = agg([r["recall_15"] for r in seed_results])
        r20_m, r20_s, _ = agg([r["recall_20"] for r in seed_results])
        r25_m, r25_s, _ = agg([r["recall_25"] for r in seed_results])
        ro_entries[key] = {
            "n_seeds": n_ok,
            "mae_mean": mae_m, "mae_std": mae_s,
            "spearman_r_mean": sp_m, "spearman_r_std": sp_s,
            "top10pct_recall_mean": r10_m, "top10pct_recall_std": r10_s,
            "top15pct_recall_mean": r15_m, "top15pct_recall_std": r15_s,
            "top20pct_recall_mean": r20_m, "top20pct_recall_std": r20_s,
            "top25pct_recall_mean": r25_m, "top25pct_recall_std": r25_s,
            "per_seed": seed_results,
        }

    # Write outputs
    lc_out = OUT_DIR / "learning_curves_ext.json"
    lc_data = {
        "train_sizes": [N for N in TRAIN_SIZES if str(N) in lc_entries],
        "n_seeds": N_SEEDS,
        "n_total": 364,
        "n_test": 73,
        "learning_curve": lc_entries,
    }
    json.dump(lc_data, open(lc_out, "w"), indent=2)
    print(f"\n✅ Written: {lc_out.relative_to(PROJECT)}")

    ro_out = OUT_DIR / "rank_order_lc_ext.json"
    ro_data = {
        "train_sizes": [N for N in TRAIN_SIZES if str(N) in ro_entries],
        "learning_curve": ro_entries,
    }
    json.dump(ro_data, open(ro_out, "w"), indent=2)
    print(f"✅ Written: {ro_out.relative_to(PROJECT)}")

    # Print summary table
    print("\nSummary Table:")
    print(f"{'N':>5}  {'MAE±std':>14}  {'Spearman':>10}  {'Recall10':>10}  {'Recall20':>10}  {'n_seeds':>7}")
    print("-" * 68)
    for N in TRAIN_SIZES:
        key = str(N)
        if key in ro_entries:
            e = ro_entries[key]
            print(f"{N:>5}  {e['mae_mean']:.3f}±{e['mae_std']:.3f}      "
                  f"{e['spearman_r_mean']:.3f}       "
                  f"{e['top10pct_recall_mean']:.3f}       "
                  f"{e['top20pct_recall_mean']:.3f}       "
                  f"{e['n_seeds']:>7}")


if __name__ == "__main__":
    main()
