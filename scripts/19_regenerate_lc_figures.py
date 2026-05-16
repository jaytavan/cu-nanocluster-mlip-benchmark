"""
19_regenerate_lc_figures.py

Regenerate Fig 9 (fig8_learning_curves.png) and Fig 11 (fig10_rank_order_lc.png)
using the 364-site extended dataset results.

Sources:
  results/learning_curves_ext.json   — MACE MAE LC (364-site, N=10-100)
  results/chgnet_ro_lc.json          — CHGNet LC + rank-order (364-site, N=10-80)
  results/rank_order_lc_ext.json     — MACE rank-order LC (364-site, N=10-100)
  results/benchmark_summary.json     — zero-shot MAE baselines
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "results"
VIZ     = PROJECT / "viz"

# ── palette ──────────────────────────────────────────────────────────────────
C_MACE_FT  = "#e74c3c"   # fine-tuned MACE — red
C_CHGNET_FT= "#27ae60"   # fine-tuned CHGNet — green
C_MACE_ZS  = "#3498db"   # MACE zero-shot — blue
C_CHGNET_ZS= "#95a5a6"   # CHGNet zero-shot — grey
C_TN_ZS    = "#9b59b6"   # TensorNet zero-shot — purple
GREY_FILL  = 0.15        # fill alpha


def load_data():
    lc_mace  = json.load(open(RESULTS / "learning_curves_ext.json"))["learning_curve"]
    ro_mace  = json.load(open(RESULTS / "rank_order_lc_ext.json"))["learning_curve"]
    chg      = json.load(open(RESULTS / "chgnet_ro_lc.json"))
    lc_chg   = chg["learning_curve"]
    zs_chg   = chg["zero_shot"]
    bs       = json.load(open(RESULTS / "benchmark_summary.json"))["by_model"]
    return lc_mace, ro_mace, lc_chg, zs_chg, bs


def fig9_learning_curves(lc_mace, lc_chg, bs):
    """Fig 9 — MAE learning curves for MACE + CHGNet (364-site dataset)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # ── MACE fine-tuned curve ─────────────────────────────────────────────
    mace_Ns  = sorted(int(k) for k in lc_mace)
    mace_mae = [lc_mace[str(N)]["mae_mean"] for N in mace_Ns]
    mace_std = [lc_mace[str(N)]["mae_std"]  for N in mace_Ns]
    ax.plot(mace_Ns, mace_mae, "o-", color=C_MACE_FT, lw=2.5, ms=7,
            label="MACE-MP-0 (fine-tuned)", zorder=4)
    ax.fill_between(mace_Ns,
                    [m - s for m, s in zip(mace_mae, mace_std)],
                    [m + s for m, s in zip(mace_mae, mace_std)],
                    alpha=GREY_FILL, color=C_MACE_FT)

    # ── CHGNet fine-tuned curve ───────────────────────────────────────────
    chg_Ns  = sorted(int(k) for k in lc_chg)
    chg_mae = [lc_chg[str(N)]["mae_mean"] for N in chg_Ns]
    chg_std = [lc_chg[str(N)]["mae_std"]  for N in chg_Ns]
    ax.plot(chg_Ns, chg_mae, "s-", color=C_CHGNET_FT, lw=2.5, ms=7,
            label="CHGNet (fine-tuned)", zorder=4)
    ax.fill_between(chg_Ns,
                    [m - s for m, s in zip(chg_mae, chg_std)],
                    [m + s for m, s in zip(chg_mae, chg_std)],
                    alpha=GREY_FILL, color=C_CHGNET_FT)

    # ── Zero-shot baselines ───────────────────────────────────────────────
    max_N = max(max(mace_Ns), max(chg_Ns))
    ax.axhline(bs["MACE-MP-0"]["mae_eV"],  ls="--", color=C_MACE_ZS,   lw=1.6,
               label=f"MACE-MP-0 zero-shot ({bs['MACE-MP-0']['mae_eV']:.3f} eV)")
    ax.axhline(bs["CHGNet"]["mae_eV"],     ls="-.", color=C_CHGNET_ZS,  lw=1.6,
               label=f"CHGNet zero-shot ({bs['CHGNet']['mae_eV']:.3f} eV)")
    ax.axhline(bs["TensorNet"]["mae_eV"],  ls=":",  color=C_TN_ZS,      lw=1.6,
               label=f"TensorNet zero-shot ({bs['TensorNet']['mae_eV']:.3f} eV)")

    # ── Annotation: convergence ───────────────────────────────────────────
    ax.annotate("0.087 eV\n(N = 100)",
                xy=(100, lc_mace["100"]["mae_mean"]),
                xytext=(85, 0.18),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=C_MACE_FT, lw=1.2),
                color=C_MACE_FT)

    ax.set_xlabel("Fine-tuning dataset size N (DFT calculations)", fontsize=12)
    ax.set_ylabel("MAE (eV)", fontsize=12)
    ax.set_title("Fine-Tuning MAE Learning Curves\n"
                 r"Cu$_{10}$–Cu$_{50}$, 364-site dataset, test set $n$ = 73",
                 fontsize=11)
    ax.set_xlim(left=0, right=max_N + 5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.3, ls=":")

    plt.tight_layout()
    out = VIZ / "fig8_learning_curves.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Fig 9 → {out.relative_to(PROJECT)}")
    return out


def fig11_rank_order_lc(ro_mace, lc_chg, zs_chg):
    """Fig 11 — Rank-order fidelity LC: Spearman ρ + top-20% recall."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"Rank-Order Fidelity vs. Fine-Tuning Dataset Size"
        "\n"
        r"MACE-MP-0 fine-tuned — Cu$_{10}$–Cu$_{50}$, 364-site dataset",
        fontsize=12, fontweight="bold",
    )

    mace_Ns = sorted(int(k) for k in ro_mace)
    zs_models = {
        "MACE-MP-0 (zero-shot)": (zs_chg["MACE-MP-0"], C_MACE_ZS, "--"),
        "CHGNet (zero-shot)":    (zs_chg["CHGNet"],     C_CHGNET_ZS, "-."),
        "TensorNet (zero-shot)": (zs_chg["TensorNet"],  C_TN_ZS,    ":"),
    }

    metrics = [
        ("spearman_r_mean",      "spearman_r_std",
         "spearman_r",           "Spearman ρ",
         "Rank Correlation (Spearman ρ)", (0, 1)),
        ("top20pct_recall_mean", "top20pct_recall_std",
         "top20pct_recall",      "Top-20% Recall",
         "Top-20% Screening Recall", (0, 1)),
    ]

    for ax, (mean_k, std_k, zs_k, ylabel, title, ylim) in zip(axes, metrics):
        means = [ro_mace[str(N)][mean_k] for N in mace_Ns]
        stds  = [ro_mace[str(N)][std_k]  for N in mace_Ns]

        ax.plot(mace_Ns, means, "o-", color=C_MACE_FT, lw=2.5, ms=8,
                label="MACE-MP-0 (fine-tuned)", zorder=4)
        ax.fill_between(mace_Ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=GREY_FILL, color=C_MACE_FT)

        # CHGNet fine-tuned on same plot
        chg_Ns  = sorted(int(k) for k in lc_chg)
        chg_m   = [lc_chg[str(N)].get(mean_k, 0) for N in chg_Ns]
        chg_s   = [lc_chg[str(N)].get(std_k,  0) for N in chg_Ns]
        if any(v > 0 for v in chg_m):
            ax.plot(chg_Ns, chg_m, "s-", color=C_CHGNET_FT, lw=2.0, ms=7,
                    label="CHGNet (fine-tuned)", zorder=3)
            ax.fill_between(chg_Ns,
                            [m - s for m, s in zip(chg_m, chg_s)],
                            [m + s for m, s in zip(chg_m, chg_s)],
                            alpha=GREY_FILL, color=C_CHGNET_FT)

        # Zero-shot baselines
        for label, (zs_data, color, ls) in zs_models.items():
            val = zs_data.get(zs_k)
            if val is not None:
                ax.axhline(val, ls=ls, color=color, lw=1.8, label=label, zorder=2)

        ax.set_xlabel("Fine-tuning dataset size N (DFT calculations)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(left=0, right=max(mace_Ns) + 5)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9, framealpha=0.92)
        ax.grid(True, alpha=0.3, ls=":")

    plt.tight_layout()
    out = VIZ / "fig10_rank_order_lc.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Fig 11 → {out.relative_to(PROJECT)}")
    return out


def main():
    print("=" * 60)
    print("Script 19 — Regenerate LC Figures (364-site extended data)")
    print("=" * 60)

    lc_mace, ro_mace, lc_chg, zs_chg, bs = load_data()

    print("\n[1] Fig 9 — MAE learning curves...")
    fig9_learning_curves(lc_mace, lc_chg, bs)

    print("\n[2] Fig 11 — Rank-order LC...")
    fig11_rank_order_lc(ro_mace, lc_chg, zs_chg)

    print("\nDone.")


if __name__ == "__main__":
    main()
