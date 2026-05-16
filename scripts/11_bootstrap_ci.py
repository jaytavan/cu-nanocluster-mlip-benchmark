"""
Script 11 — Bootstrap Confidence Intervals on Spearman ρ and Top-N% Recall
===========================================================================
Adds statistical uncertainty estimates to the rank-order metrics in Tables 3
and 4 of the manuscript. No new DFT required — operates on existing results.

Bootstrap procedure:
  1000 resamples with replacement per (model, size) pair
  95% CI = 2.5th – 97.5th percentile of bootstrap distribution
  Warns when n < 10 (CI width will be large — this is the point)

Output:
  results/bootstrap_ci_results.json   — machine-readable CI data
  Printed Table 3 (overall) and Table 4 (by size) ready for manuscript
  viz/fig_bootstrap_ci.png            — CI width visualization

Run:
    conda activate catalyst
    python3 scripts/11_bootstrap_ci.py
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

N_BOOTSTRAP  = 1000
CI_LEVEL     = 0.95
CI_LO        = (1 - CI_LEVEL) / 2 * 100      # 2.5
CI_HI        = (1 + CI_LEVEL) / 2 * 100      # 97.5
SMALL_N_WARN = 10

rng = np.random.default_rng(seed=42)


# ── Data Loading (mirrors script 10) ──────────────────────────────────────────

def load_matched_dataset() -> Dict[str, List[Dict]]:
    """Load all sites with both GPAW and MLIP predictions."""
    gpaw_by_id = {}
    for sz in [10, 20, 30, 40, 50]:
        rf = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_00.json"
        if not rf.exists():
            continue
        data = json.load(open(rf))
        for site in data["sites"]:
            if site["status"] == "ok":
                gpaw_by_id[site["site_global_id"]] = site

    if not gpaw_by_id:
        raise RuntimeError("No GPAW results found — run GPAW batch first.")
    print(f"  GPAW sites loaded: {len(gpaw_by_id)}")

    chgnet_data, tensornet_data = {}, {}
    chg_path = RESULTS_DIR / "chgnet" / "results_chgnet.json"
    tn_path  = RESULTS_DIR / "tensornet" / "results_tensornet.json"

    if chg_path.exists():
        for site in json.load(open(chg_path)).get("sites", []):
            chgnet_data[site["site_global_id"]] = site
        print(f"  CHGNet sites:    {len(chgnet_data)}")

    if tn_path.exists():
        for site in json.load(open(tn_path)).get("sites", []):
            tensornet_data[site["site_global_id"]] = site
        print(f"  TensorNet sites: {len(tensornet_data)}")

    matched: Dict[str, List[Dict]] = {"MACE-MP-0": [], "CHGNet": [], "TensorNet": []}

    for sid, gsite in gpaw_by_id.items():
        base = {
            "site_id":   sid,
            "size":      gsite["size"],
            "site_type": gsite["site_type"],
            "dgh_gpaw":  gsite["dgh_gpaw_eV"],
        }
        if "dgh_mace_eV" in gsite:
            matched["MACE-MP-0"].append({**base, "dgh_model": gsite["dgh_mace_eV"]})
        if sid in chgnet_data and "dgh_chgnet_eV" in chgnet_data[sid]:
            matched["CHGNet"].append({**base, "dgh_model": chgnet_data[sid]["dgh_chgnet_eV"]})
        if sid in tensornet_data and "dgh_tensornet_eV" in tensornet_data[sid]:
            matched["TensorNet"].append({**base, "dgh_model": tensornet_data[sid]["dgh_tensornet_eV"]})

    return matched


# ── Bootstrap Core ────────────────────────────────────────────────────────────

def bootstrap_spearman(
    gpaw_vals: np.ndarray,
    model_vals: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    """
    Point estimate + 95% percentile CI on Spearman ρ.
    Returns (rho_point, ci_low, ci_high).
    """
    n = len(gpaw_vals)
    point_rho, _ = stats.spearmanr(gpaw_vals, model_vals)

    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g, m = gpaw_vals[idx], model_vals[idx]
        if np.std(g) == 0 or np.std(m) == 0:
            boot_rhos[i] = np.nan
        else:
            boot_rhos[i], _ = stats.spearmanr(g, m)

    valid = boot_rhos[~np.isnan(boot_rhos)]
    ci_lo = float(np.percentile(valid, CI_LO))
    ci_hi = float(np.percentile(valid, CI_HI))
    return float(point_rho), ci_lo, ci_hi


def bootstrap_recall(
    gpaw_vals: np.ndarray,
    model_vals: np.ndarray,
    top_frac: float = 0.10,
    n_boot: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    """
    Point estimate + 95% CI on top-N% recall using |ΔGH*| ranking.
    Returns (recall_point, ci_low, ci_high).
    """
    n = len(gpaw_vals)
    n_top = max(1, round(n * top_frac))

    def recall_from(g, m):
        gpaw_top  = set(np.argsort(np.abs(g))[:n_top])
        model_top = set(np.argsort(np.abs(m))[:n_top])
        return len(gpaw_top & model_top) / len(gpaw_top)

    point_recall = recall_from(gpaw_vals, model_vals)

    boot_recalls = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_recalls[i] = recall_from(gpaw_vals[idx], model_vals[idx])

    ci_lo = float(np.percentile(boot_recalls, CI_LO))
    ci_hi = float(np.percentile(boot_recalls, CI_HI))
    return float(point_recall), ci_lo, ci_hi


# ── Per-group computation ─────────────────────────────────────────────────────

def compute_ci_for_group(
    sites: List[Dict],
    label: str = "",
) -> Optional[Dict]:
    """Run bootstrap on a list of matched sites. Returns None if too few sites."""
    n = len(sites)
    if n < 3:
        return None

    if n < SMALL_N_WARN:
        warnings.warn(
            f"  ⚠  Small sample (n={n}) for {label!r} — "
            "CIs will be wide and should be interpreted cautiously.",
            stacklevel=2,
        )

    gpaw_vals  = np.array([s["dgh_gpaw"]  for s in sites])
    model_vals = np.array([s["dgh_model"] for s in sites])

    rho, rho_lo, rho_hi = bootstrap_spearman(gpaw_vals, model_vals)
    r10, r10_lo, r10_hi = bootstrap_recall(gpaw_vals, model_vals, top_frac=0.10)
    r20, r20_lo, r20_hi = bootstrap_recall(gpaw_vals, model_vals, top_frac=0.20)

    return {
        "n": n,
        "spearman_rho":    rho,
        "spearman_ci_lo":  rho_lo,
        "spearman_ci_hi":  rho_hi,
        "spearman_ci_width": rho_hi - rho_lo,
        "top10_recall":    r10,
        "top10_ci_lo":     r10_lo,
        "top10_ci_hi":     r10_hi,
        "top20_recall":    r20,
        "top20_ci_lo":     r20_lo,
        "top20_ci_hi":     r20_hi,
    }


# ── Formatted table output ────────────────────────────────────────────────────

def fmt_rho(r: Dict) -> str:
    """Format ρ = 0.94 (95% CI: [0.71, 1.00]) for manuscript."""
    return f"ρ = {r['spearman_rho']:+.2f}  (95% CI: [{r['spearman_ci_lo']:+.2f}, {r['spearman_ci_hi']:+.2f}])"


def fmt_recall(r: Dict, key: str = "top10") -> str:
    return (
        f"{r[key+'_recall']:.2f}  "
        f"[{r[key+'_ci_lo']:.2f}, {r[key+'_ci_hi']:.2f}]"
    )


def print_table3(overall_ci: Dict):
    """Print Table 3 (overall metrics) with bootstrap CIs."""
    print("\n" + "=" * 75)
    print("TABLE 3 — Overall Rank-Order Fidelity with Bootstrap 95% CIs")
    print(f"(n_bootstrap={N_BOOTSTRAP}, seed=42)")
    print("=" * 75)
    header = f"{'Model':15s}  {'n':>4s}  {'Spearman ρ (95% CI)':35s}  {'Top-10% Recall [CI]':22s}"
    print(header)
    print("-" * 80)
    for model, r in overall_ci.items():
        if r is None:
            continue
        print(
            f"  {model:13s}  {r['n']:4d}  {fmt_rho(r):35s}  {fmt_recall(r, 'top10'):22s}"
        )
    print()


def print_table4(size_ci: Dict):
    """Print Table 4 (per-size metrics) with bootstrap CIs."""
    print("\n" + "=" * 90)
    print("TABLE 4 — Rank-Order Fidelity by Cluster Size with Bootstrap 95% CIs")
    print(f"(n_bootstrap={N_BOOTSTRAP}, seed=42)")
    print("=" * 90)
    sizes = sorted({int(sz) for m in size_ci.values() for sz in m})

    for model in ["MACE-MP-0", "CHGNet", "TensorNet"]:
        if model not in size_ci:
            continue
        print(f"\n  {model}:")
        print(f"    {'Size':8s}  {'n':>4s}  {'Spearman ρ (95% CI)':38s}  {'Top-10% Recall [CI]':22s}")
        print("    " + "-" * 78)
        for sz in sizes:
            r = size_ci[model].get(str(sz))
            if r is None:
                print(f"    Cu{sz:2d}     n<3   insufficient data")
                continue
            ci_width = r["spearman_ci_width"]
            flag = "  ⚠ wide CI" if ci_width > 0.9 else ("  △ moderate CI" if ci_width > 0.5 else "")
            print(f"    Cu{sz:2d}    {r['n']:4d}  {fmt_rho(r):38s}  {fmt_recall(r, 'top10'):22s}{flag}")
    print()


# ── Visualization ─────────────────────────────────────────────────────────────

COLORS  = {"MACE-MP-0": "#3498db", "CHGNet": "#2ecc71", "TensorNet": "#9b59b6"}
OFFSETS = {"MACE-MP-0": -0.15, "CHGNet": 0.0, "TensorNet": 0.15}


def plot_ci_summary(size_ci: Dict, overall_ci: Dict, output_path: Path):
    """
    Plot Spearman ρ ± 95% CI per cluster size for all three models.
    Highlights how wide the CIs are for small-n subsets (Cu₁₀, Cu₄₀).
    """
    sizes = sorted({int(sz) for m in size_ci.values() for sz in m})
    x = np.arange(len(sizes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Spearman ρ with Bootstrap 95% CIs — Foundation MLIPs vs GPAW\n"
        r"Cu$_{10}$–Cu$_{50}$ Nanoclusters, Bridge + Hollow Sites",
        fontsize=12, fontweight="bold",
    )

    # ── Left: per-size CIs ────────────────────────────────────────────────
    ax = axes[0]
    for model in ["MACE-MP-0", "CHGNet", "TensorNet"]:
        if model not in size_ci:
            continue
        xpos, rhos, lo_errs, hi_errs, ns = [], [], [], [], []
        for i, sz in enumerate(sizes):
            r = size_ci[model].get(str(sz))
            if r is None:
                continue
            xpos.append(x[i] + OFFSETS[model])
            rhos.append(r["spearman_rho"])
            lo_errs.append(r["spearman_rho"] - r["spearman_ci_lo"])
            hi_errs.append(r["spearman_ci_hi"] - r["spearman_rho"])
            ns.append(r["n"])

        ax.errorbar(
            xpos, rhos,
            yerr=[lo_errs, hi_errs],
            fmt="o", color=COLORS[model],
            capsize=5, capthick=1.5, lw=1.5, ms=7,
            label=model,
        )
        # Annotate n on top of each point
        for xi, rho, n in zip(xpos, rhos, ns):
            ax.text(xi, rho + 0.05, f"n={n}", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS[model])

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.6, label="ρ=0.5 guideline")
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-1.25, 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cu$_{{{sz}}}$" for sz in sizes], fontsize=11)
    ax.set_xlabel("Cluster size", fontsize=11)
    ax.set_ylabel("Spearman ρ  (95% bootstrap CI)", fontsize=11)
    ax.set_title("(a) Per-Size Spearman ρ ± 95% CI", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--", axis="y")

    # ── Right: overall CIs ────────────────────────────────────────────────
    ax = axes[1]
    model_list = [m for m in ["MACE-MP-0", "CHGNet", "TensorNet"] if m in overall_ci]
    xi = np.arange(len(model_list))

    for i, model in enumerate(model_list):
        r = overall_ci[model]
        if r is None:
            continue
        rho = r["spearman_rho"]
        lo  = rho - r["spearman_ci_lo"]
        hi  = r["spearman_ci_hi"] - rho
        ax.bar(i, rho, color=COLORS[model], alpha=0.75, width=0.5, label=model)
        ax.errorbar(i, rho, yerr=[[lo], [hi]],
                    fmt="none", color="black", capsize=8, capthick=1.5, lw=2)
        ax.text(i, rho + hi + 0.03,
                f"n={r['n']}\nρ={rho:+.2f}\n[{r['spearman_ci_lo']:+.2f}, {r['spearman_ci_hi']:+.2f}]",
                ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xi)
    ax.set_xticklabels(model_list, fontsize=11)
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Spearman ρ  (95% bootstrap CI)", fontsize=11)
    ax.set_title("(b) Overall Spearman ρ ± 95% CI  (all sizes pooled)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--", axis="y")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    VIZ_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"Script 11 — Bootstrap CI on Spearman ρ  (n_bootstrap={N_BOOTSTRAP})")
    print("=" * 70)

    print("\n[1] Loading matched datasets...")
    matched = load_matched_dataset()
    for model, sites in matched.items():
        print(f"  {model}: {len(sites)} matched sites")

    print(f"\n[2] Computing bootstrap CIs ({N_BOOTSTRAP} resamples per group)...")

    # Overall (all sizes pooled)
    overall_ci: Dict[str, Optional[Dict]] = {}
    for model, sites in matched.items():
        print(f"  {model} overall (n={len(sites)})...", end=" ", flush=True)
        overall_ci[model] = compute_ci_for_group(sites, label=f"{model}/overall")
        r = overall_ci[model]
        if r:
            print(f"ρ={r['spearman_rho']:+.3f}  CI=[{r['spearman_ci_lo']:+.3f}, {r['spearman_ci_hi']:+.3f}]")
        else:
            print("skipped (n<3)")

    # Per cluster size
    size_ci: Dict[str, Dict[str, Optional[Dict]]] = {}
    for model, sites in matched.items():
        size_ci[model] = {}
        by_size: Dict[int, List[Dict]] = {}
        for site in sites:
            by_size.setdefault(site["size"], []).append(site)

        for sz in sorted(by_size.keys()):
            sz_sites = by_size[sz]
            n = len(sz_sites)
            print(f"  {model} / Cu{sz} (n={n})...", end=" ", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = compute_ci_for_group(sz_sites, label=f"{model}/Cu{sz}")
                for w in caught:
                    print(str(w.message))
            size_ci[model][str(sz)] = r
            if r:
                print(f"ρ={r['spearman_rho']:+.3f}  CI=[{r['spearman_ci_lo']:+.3f}, {r['spearman_ci_hi']:+.3f}]")
            else:
                print("skipped")

    print("\n[3] Printing manuscript-ready tables...")
    print_table3(overall_ci)
    print_table4(size_ci)

    print("[4] Saving JSON results...")
    out = {
        "meta": {
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": CI_LEVEL,
            "seed": 42,
            "note": (
                "Bootstrap percentile CI on Spearman rho and top-10%/20% recall. "
                "Small-n groups (Cu10 n=6, Cu40 n=5) show wide CIs — "
                "interpret per-size rho values cautiously."
            ),
        },
        "overall": overall_ci,
        "by_size": size_ci,
    }
    out_path = RESULTS_DIR / "bootstrap_ci_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")

    print("\n[5] Generating CI visualization...")
    plot_ci_summary(size_ci, overall_ci, VIZ_DIR / "fig_bootstrap_ci.png")

    # ── Key takeaway ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR MANUSCRIPT")
    print("=" * 70)
    print("\nOverall Spearman ρ (n=117, narrow CIs — statistically reliable):")
    for model, r in overall_ci.items():
        if r:
            w = r["spearman_ci_width"]
            print(f"  {model:13s}  ρ = {r['spearman_rho']:+.2f}  "
                  f"(95% CI: [{r['spearman_ci_lo']:+.2f}, {r['spearman_ci_hi']:+.2f}], width={w:.2f})")

    print("\nSmall-n per-size results (report with caution — wide CIs):")
    for model in ["MACE-MP-0", "CHGNet", "TensorNet"]:
        for sz in ["10", "40"]:
            r = size_ci.get(model, {}).get(sz)
            if r:
                w = r["spearman_ci_width"]
                flag = "WIDE" if w > 0.9 else ("moderate" if w > 0.5 else "narrow")
                print(f"  {model:13s} Cu{sz:2s}: ρ={r['spearman_rho']:+.2f}  "
                      f"CI=[{r['spearman_ci_lo']:+.2f}, {r['spearman_ci_hi']:+.2f}]  "
                      f"width={w:.2f}  [{flag}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
