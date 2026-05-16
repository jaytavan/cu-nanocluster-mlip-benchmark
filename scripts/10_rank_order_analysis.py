"""
Script 10 — Rank-Order Fidelity Analysis
=========================================
Computes Spearman rank correlation and top-N% recall for each MLIP model
compared to GPAW reference rankings.

This is the screening-quality metric: do the ML models identify the right
candidate sites for HER, even if their absolute ΔGH* values are off?

Key output metrics:
  - Spearman ρ: rank correlation between MLIP and GPAW ΔGH* rankings
  - Top-10% recall: fraction of GPAW top-10% sites captured by MLIP top-10%
  - Top-20% recall: same with 20% threshold
  - False-negative analysis: which sites does ML miss and why?

Generates:
  - fig9_rank_order_fidelity.png  (scatter + recall curves)
  - results/rank_order_results.json

Run after script 07 (needs benchmark_summary.json and GPAW results):
    python3 scripts/10_rank_order_analysis.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
GPAW_DIR    = RESULTS_DIR / "gpaw"
VIZ_DIR     = PROJECT_DIR / "viz"

ZPE_CORR = 0.24  # eV


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_matched_dataset() -> Dict[str, List[Dict]]:
    """
    Load all sites with both GPAW and MLIP predictions.
    Returns: {"MACE-MP-0": [...], "CHGNet": [...], "TensorNet": [...]}
    Each entry: {"site_id", "size", "site_type", "dgh_gpaw", "dgh_model"}
    """
    # Load GPAW results (all cluster IDs: _00, _01, _02)
    gpaw_by_id = {}
    for sz in [10, 20, 30, 40, 50]:
        for cid_suffix in ['00', '01', '02']:
            rf = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_{cid_suffix}.json"
            if not rf.exists():
                continue
            data = json.load(open(rf))
            for site in data["sites"]:
                if site["status"] == "ok":
                    gpaw_by_id[site["site_global_id"]] = site

    if not gpaw_by_id:
        raise RuntimeError("No GPAW results found. Run GPAW batch first.")

    print(f"  GPAW sites available: {len(gpaw_by_id)}")

    # Load CHGNet results
    chgnet_data, tensornet_data = {}, {}
    chg_path = RESULTS_DIR / "chgnet" / "results_chgnet.json"
    tn_path  = RESULTS_DIR / "tensornet" / "results_tensornet.json"

    if chg_path.exists():
        chg = json.load(open(chg_path))
        for site in chg.get("sites", []):
            chgnet_data[site["site_global_id"]] = site
        print(f"  CHGNet sites: {len(chgnet_data)}")

    if tn_path.exists():
        tn = json.load(open(tn_path))
        for site in tn.get("sites", []):
            tensornet_data[site["site_global_id"]] = site
        print(f"  TensorNet sites: {len(tensornet_data)}")

    # Build matched datasets
    matched = {"MACE-MP-0": [], "CHGNet": [], "TensorNet": []}

    for sid, gsite in gpaw_by_id.items():
        base = {
            "site_id":   sid,
            "size":      gsite["size"],
            "site_type": gsite["site_type"],
            "dgh_gpaw":  gsite["dgh_gpaw_eV"],
        }

        # MACE: from GPAW site records (dgh_mace_eV stored there)
        if "dgh_mace_eV" in gsite:
            matched["MACE-MP-0"].append({**base, "dgh_model": gsite["dgh_mace_eV"]})

        # CHGNet
        if sid in chgnet_data and "dgh_chgnet_eV" in chgnet_data[sid]:
            matched["CHGNet"].append({**base, "dgh_model": chgnet_data[sid]["dgh_chgnet_eV"]})

        # TensorNet
        if sid in tensornet_data and "dgh_tensornet_eV" in tensornet_data[sid]:
            matched["TensorNet"].append({**base, "dgh_model": tensornet_data[sid]["dgh_tensornet_eV"]})

    return matched


# ── Rank-Order Metrics ────────────────────────────────────────────────────────

def compute_rank_metrics(sites: List[Dict], top_fractions: List[float] = None) -> Dict:
    """
    Compute Spearman ρ and top-N% recall for a matched site list.

    top_fractions: list of fractions (e.g., [0.10, 0.20, 0.30])
    """
    if top_fractions is None:
        top_fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    gpaw_vals  = np.array([s["dgh_gpaw"] for s in sites])
    model_vals = np.array([s["dgh_model"] for s in sites])

    # Spearman ρ
    spearman_r, spearman_p = stats.spearmanr(gpaw_vals, model_vals)

    # Pearson r
    pearson_r = float(np.corrcoef(gpaw_vals, model_vals)[0, 1])

    # Top-N% recall
    n_total = len(sites)
    recalls = {}
    for frac in top_fractions:
        n_top = max(1, round(n_total * frac))

        # GPAW top sites (lowest ΔGH* closest to 0 — best HER candidates)
        # For HER, optimal ΔGH* ~ 0 eV. Use |ΔGH*| as proxy for ranking.
        gpaw_rank  = np.argsort(np.abs(gpaw_vals))[:n_top]
        model_rank = np.argsort(np.abs(model_vals))[:n_top]

        true_top  = set(gpaw_rank)
        pred_top  = set(model_rank)
        recall    = len(true_top & pred_top) / len(true_top)
        recalls[f"top{int(frac*100)}pct_recall"] = float(recall)

    return {
        "n_sites":     n_total,
        "spearman_r":  float(spearman_r),
        "spearman_p":  float(spearman_p),
        "pearson_r":   pearson_r,
        "mae_eV":      float(np.mean(np.abs(gpaw_vals - model_vals))),
        "rmse_eV":     float(np.sqrt(np.mean((gpaw_vals - model_vals) ** 2))),
        **recalls,
    }


def compute_by_size(sites: List[Dict]) -> Dict[int, Dict]:
    """Compute rank metrics per cluster size."""
    by_size = {}
    for site in sites:
        by_size.setdefault(site["size"], []).append(site)
    return {sz: compute_rank_metrics(ss) for sz, ss in sorted(by_size.items()) if len(ss) >= 3}


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "MACE-MP-0": "#3498db",
    "CHGNet":    "#2ecc71",
    "TensorNet": "#9b59b6",
}
MARKERS = {"MACE-MP-0": "o", "CHGNet": "s", "TensorNet": "^"}


def plot_rank_fidelity(matched: Dict, metrics: Dict, output_path: Path):
    """
    Generate Fig 9: rank-order fidelity analysis.

    3-panel figure (horizontal):
    (a) Spearman ρ per cluster size
    (b) Top-N% recall curves (recall vs N_top% threshold)
    (c) False negative: GPAW top-10% sites missed by each model
    """
    fig, (ax_spearman, ax_recall, ax_fn) = plt.subplots(
        1, 3, figsize=(15, 5.5)
    )
    fig.subplots_adjust(wspace=0.34, left=0.06, right=0.97, top=0.86, bottom=0.13)

    fig.suptitle(
        "Zero-Shot Rank-Order Fidelity: Foundation MLIPs vs GPAW for HER Site Ranking\n"
        r"Cu$_{10}$–Cu$_{50}$ Nanoclusters",
        fontsize=13, fontweight="bold",
    )

    # ── (a) Spearman ρ per cluster size ─────────────────────────────────
    ax = ax_spearman
    sizes = sorted(set(s["size"] for v in matched.values() for s in v))
    x = np.arange(len(sizes))
    bar_w = 0.25

    for i, (model_name, sites) in enumerate(matched.items()):
        if not sites:
            continue
        size_metrics = compute_by_size(sites)
        rhos = [size_metrics.get(sz, {}).get("spearman_r", np.nan) for sz in sizes]
        ax.bar(x + (i - 1) * bar_w, rhos, width=bar_w,
               color=COLORS[model_name], alpha=0.8, label=model_name)

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xlabel("Cluster size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title("(a) Spearman ρ by Cluster Size", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cu$_{{{sz}}}$" for sz in sizes], fontsize=10)
    ax.set_ylim(-1, 1)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--", axis="y")

    # ── (c) Top-N% recall curves ─────────────────────────────────────────
    ax = ax_recall
    fracs = np.arange(0.05, 0.55, 0.05)

    for model_name, sites in matched.items():
        if not sites:
            continue
        gpaw_vals  = np.array([s["dgh_gpaw"] for s in sites])
        model_vals = np.array([s["dgh_model"] for s in sites])
        n_total    = len(sites)

        recalls = []
        for frac in fracs:
            n_top = max(1, round(n_total * frac))
            gpaw_top  = set(np.argsort(np.abs(gpaw_vals))[:n_top])
            model_top = set(np.argsort(np.abs(model_vals))[:n_top])
            recalls.append(len(gpaw_top & model_top) / len(gpaw_top))

        ax.plot(fracs * 100, recalls, "o-",
                color=COLORS[model_name], marker=MARKERS[model_name],
                ms=5, lw=2, label=model_name)

    # Perfect recall diagonal
    ax.plot(fracs * 100, fracs * 100 / 100 + (1 - fracs * 100 / 100) * 0,
            "k--", lw=1, alpha=0.4)
    ax.plot([5, 50], [5/100 * 1.0, 50/100 * 1.0], "k:", lw=1,  # random baseline
            alpha=0.5, label="Random baseline")
    ax.set_xlabel("Top-N% threshold", fontsize=11)
    ax.set_ylabel("Recall (fraction of true top sites found)", fontsize=11)
    ax.set_title("(b) Top-N% Screening Recall", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")

    # ── (d) False negative: error distribution of missed top sites ───────
    ax = ax_fn
    TOP_FRAC = 0.10  # focus on top 10%

    for model_name, sites in matched.items():
        if not sites:
            continue
        gpaw_vals  = np.array([s["dgh_gpaw"] for s in sites])
        model_vals = np.array([s["dgh_model"] for s in sites])
        n_total    = len(sites)
        n_top      = max(1, round(n_total * TOP_FRAC))

        gpaw_top_idx   = set(np.argsort(np.abs(gpaw_vals))[:n_top])
        model_top_idx  = set(np.argsort(np.abs(model_vals))[:n_top])
        false_neg_idx  = gpaw_top_idx - model_top_idx  # missed by model

        if false_neg_idx:
            fn_gpaw  = np.array([gpaw_vals[i] for i in false_neg_idx])
            fn_model = np.array([model_vals[i] for i in false_neg_idx])
            fn_error = fn_model - fn_gpaw

            ax.scatter(fn_gpaw, fn_model,
                       color=COLORS[model_name], marker=MARKERS[model_name],
                       alpha=0.7, s=40, label=f"{model_name} (n={len(fn_gpaw)})")

    # Optimal band
    ax.axhline(-0.3, color="green", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(+0.3, color="green", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(-0.3, color="green", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(+0.3, color="green", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.3)
    ax.axvline(0, color="black", lw=0.8, alpha=0.3)
    ax.set_xlabel(r"GPAW ΔG$_H^*$ (eV)", fontsize=11)
    ax.set_ylabel(r"MLIP ΔG$_H^*$ (eV)", fontsize=11)
    ax.set_title(f"(c) Missed Top-{int(TOP_FRAC*100)}% Sites: MLIP vs GPAW ΔG$_H^*$", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    VIZ_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 10 — Rank-Order Fidelity Analysis")
    print("=" * 65)

    print("\n[1] Loading matched datasets...")
    matched = load_matched_dataset()
    for model_name, sites in matched.items():
        print(f"  {model_name}: {len(sites)} matched sites")

    print("\n[2] Computing rank-order metrics...")
    all_metrics = {}
    for model_name, sites in matched.items():
        if not sites:
            print(f"  {model_name}: no data")
            continue
        m = compute_rank_metrics(sites)
        all_metrics[model_name] = m
        print(f"\n  {model_name} (n={m['n_sites']}):")
        print(f"    Spearman ρ = {m['spearman_r']:.3f} (p={m['spearman_p']:.3e})")
        print(f"    Pearson  r = {m['pearson_r']:.3f}")
        print(f"    MAE        = {m['mae_eV']:.4f} eV")
        print(f"    Top-10% recall = {m.get('top10pct_recall', 'n/a'):.2f}")
        print(f"    Top-20% recall = {m.get('top20pct_recall', 'n/a'):.2f}")

    print("\n[3] Computing metrics per cluster size...")
    size_metrics = {}
    for model_name, sites in matched.items():
        if not sites:
            continue
        size_metrics[model_name] = compute_by_size(sites)
        print(f"\n  {model_name} by size:")
        for sz, sm in size_metrics[model_name].items():
            print(f"    Cu{sz}: ρ={sm['spearman_r']:.2f}, "
                  f"top-10%={sm.get('top10pct_recall', 0):.2f}, "
                  f"MAE={sm['mae_eV']:.3f} eV")

    print("\n[4] Saving rank-order results...")
    out_path = RESULTS_DIR / "rank_order_results.json"
    json.dump({
        "overall":    all_metrics,
        "by_size":    {k: {str(sz): v for sz, v in vm.items()}
                       for k, vm in size_metrics.items()},
    }, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")

    print("\n[5] Generating Fig 9 — Rank-Order Fidelity...")
    plot_rank_fidelity(matched, all_metrics, VIZ_DIR / "fig9_rank_order_fidelity.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RANK-ORDER FIDELITY SUMMARY")
    print("=" * 65)
    print(f"\n{'Model':15s} {'Spearman ρ':12s} {'Top-10% recall':16s} {'MAE (eV)':10s}")
    print("-" * 55)
    for model_name, m in all_metrics.items():
        r10 = m.get("top10pct_recall", float("nan"))
        print(f"  {model_name:13s} {m['spearman_r']:+.3f}         {r10:.2f}            {m['mae_eV']:.4f}")

    print("\nKey finding: The model with highest Spearman ρ and top-10% recall is the")
    print("best choice for ML pre-screening of HER candidates.")
    print("\nDone.")


if __name__ == "__main__":
    main()
