"""
Script 14 — Cost / Accuracy Trade-off Figure (v2)
=================================================
Generates fig11: MAE vs N training points (learning curve + reference baselines).

Key design changes from v1 (backed up as 14_cost_accuracy_v1.py / fig11_cost_accuracy_v1.png):
- X-axis: N DFT training points 0→80, linear (not log wall-clock hours)
- Secondary top X-axis: approximate DFT wall-clock hours (3→24 h)
- Zero-shot models → horizontal dashed reference lines (not isolated far-left points)
- Y-axis clipped at 0.45 eV; MACE N=10 spike annotated off-scale
- Clear crossover annotation where fine-tuned MACE reaches TensorNet zero-shot accuracy

Run after scripts 09 and 13 complete:
    python3 scripts/14_cost_accuracy.py

Prerequisites:
    results/learning_curves.json       (MACE fine-tuning, from script 09)
    results/chgnet_lc.json             (CHGNet fine-tuning, from script 13)
    results/benchmark_summary.json     (zero-shot MAE values)
    results/gpaw/{Cu*}/results_*.json  (per-site GPAW timing)
    results/finetune/mace_ft_*/mace_ft_*.log  (MACE fine-tuning wall-clock times)
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
GPAW_DIR    = RESULTS_DIR / "gpaw"
VIZ_DIR     = PROJECT_DIR / "viz"

TRAIN_SIZES = [10, 20, 40, 60, 80]
TEST_FRAC   = 0.20

# ── Timing extraction (same as v1) ─────────────────────────────────────────────

def gpaw_time_per_site() -> Dict[int, float]:
    times_by_size = {}
    for sz in [10, 20, 30, 40, 50]:
        rfile = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_00.json"
        if not rfile.exists():
            continue
        data = json.load(open(rfile))
        times = [s["time_s"] for s in data["sites"]
                 if s.get("status") == "ok" and s.get("time_s")]
        if times:
            times_by_size[sz] = float(np.mean(times))
    return times_by_size


def compute_dft_budget_hr(n_train: int, train_pool: list, gpaw_times: Dict[int, float]) -> float:
    size_counts = Counter(s["size"] for s in train_pool)
    total_pool  = sum(size_counts.values())
    weighted_s  = sum(
        (count / total_pool) * gpaw_times.get(sz, np.mean(list(gpaw_times.values())))
        for sz, count in size_counts.items()
    )
    return n_train * weighted_s / 3600.0


def mace_ft_times_by_n() -> Dict[int, Tuple[float, float]]:
    ft_dir = RESULTS_DIR / "finetune"
    ts_pat = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) INFO:")
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    by_n: Dict[int, list] = {}
    for run_dir in sorted(ft_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = re.match(r"mace_ft_N(\d+)_seed\d+", run_dir.name)
        if not m:
            continue
        n = int(m.group(1))
        log = run_dir / f"{run_dir.name}.log"
        if not log.exists():
            continue
        txt = log.read_text(errors="ignore")
        timestamps = [t.group(1) for line in txt.split("\n")
                      for t in [ts_pat.search(line)] if t]
        if len(timestamps) < 2:
            continue
        elapsed = (
            datetime.strptime(timestamps[-1], fmt) -
            datetime.strptime(timestamps[0],  fmt)
        ).total_seconds()
        by_n.setdefault(n, []).append(elapsed)
    return {n: (float(np.mean(t)), float(np.std(t))) for n, t in by_n.items()}


def build_train_pool() -> list:
    all_sites = []
    for sz in [10, 20, 30, 40, 50]:
        rfile = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_00.json"
        if not rfile.exists():
            continue
        data = json.load(open(rfile))
        all_sites.extend([s for s in data["sites"] if s["status"] == "ok"])
    rng    = np.random.default_rng(seed=42)
    by_sz  = {}
    for s in all_sites:
        by_sz.setdefault(s["size"], []).append(s)
    train_pool = []
    for sz, sites in sorted(by_sz.items()):
        n_test = max(1, round(len(sites) * TEST_FRAC))
        idx    = rng.permutation(len(sites))
        train_pool.extend([sites[i] for i in idx[n_test:]])
    return train_pool


# ── Plot (v2) ──────────────────────────────────────────────────────────────────

def plot_cost_accuracy_v2(
    mace_lc:      Dict,
    chgnet_lc:    Optional[Dict],
    zero_shot:    Dict,
    mace_times:   Dict,
    gpaw_times:   Dict,
    train_pool:   list,
    output_path:  Path,
):
    # ── colours ──────────────────────────────────────────────────────────────
    C_MACE_FT  = "#e74c3c"   # red
    C_CHG_FT   = "#27ae60"   # green
    C_TN_ZS    = "#7b2d8b"   # purple
    C_MACE_ZS  = "#2980b9"   # blue
    C_CHG_ZS   = "#95a5a6"   # grey (least important baseline)

    Y_CLIP = 0.44   # clip y-axis here; MACE N=10 (0.74 eV) shown off-scale

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Zero-shot horizontal reference lines ─────────────────────────────────
    zs_mae = {m: v["mae_eV"] for m, v in zero_shot.items() if "mae_eV" in v}
    # Also try alternate key name
    if not zs_mae:
        bsum_path = RESULTS_DIR / "benchmark_summary.json"
        if bsum_path.exists():
            bsum = json.load(open(bsum_path))
            for m, v in bsum.get("by_model", {}).items():
                if "mae_eV" in v:
                    zs_mae[m] = v["mae_eV"]

    zs_styles = {
        "TensorNet":  dict(color=C_TN_ZS,   lw=1.8, ls="--", label="TensorNet zero-shot"),
        "MACE-MP-0":  dict(color=C_MACE_ZS,  lw=1.4, ls=":",  label="MACE-MP-0 zero-shot"),
        "CHGNet":     dict(color=C_CHG_ZS,   lw=1.4, ls="-.", label="CHGNet zero-shot"),
    }
    x_max = 87
    for model, mae in sorted(zs_mae.items(), key=lambda x: x[1]):
        style = zs_styles.get(model, dict(color="gray", lw=1.2, ls="--"))
        ax.axhline(mae, xmin=0, xmax=1, **{k: v for k, v in style.items() if k != "label"},
                   zorder=2, alpha=0.85)
        # Label at right edge
        ax.text(
            x_max - 1, mae + 0.007,
            style.get("label", model),
            fontsize=8, color=style["color"], ha="right", va="bottom",
        )

    # ── Fine-tuned MACE ───────────────────────────────────────────────────────
    mace_n, mace_y, mace_err = [], [], []
    mace_n10_val = None   # store for off-scale annotation

    for n in sorted(mace_lc.keys()):
        v = mace_lc[n]
        mae_mean = v.get("mae_mean", np.nan)
        mae_std  = v.get("mae_std",  0.0)
        if n == 10:
            mace_n10_val = (mae_mean, mae_std)
            # Plot clipped point + off-scale indicator instead of actual value
            ax.annotate(
                f"↑ {mae_mean:.2f} ± {mae_std:.2f} eV",
                xy=(n, Y_CLIP - 0.005),
                xytext=(n + 6, Y_CLIP - 0.06),
                fontsize=8, color=C_MACE_FT,
                arrowprops=dict(arrowstyle="->", color=C_MACE_FT, lw=1.0),
            )
            # Dashed vertical line to show it goes off-scale
            ax.plot([n, n], [Y_CLIP - 0.02, Y_CLIP], color=C_MACE_FT,
                    ls="--", lw=1.2, zorder=3)
            mace_n.append(n)
            mace_y.append(Y_CLIP - 0.01)  # clipped for the connecting line
            mace_err.append(0)
        else:
            mace_n.append(n)
            mace_y.append(mae_mean)
            mace_err.append(mae_std)
            ax.annotate(
                f"N={n}",
                xy=(n, mae_mean),
                xytext=(3, 5), textcoords="offset points",
                fontsize=7.5, color=C_MACE_FT,
            )

    # Draw line only for N>=20 (skip the clipped N=10 kink)
    plot_n  = [n for n in mace_n if n >= 20]
    plot_y  = [y for n, y in zip(mace_n, mace_y) if n >= 20]
    plot_e  = [e for n, e in zip(mace_n, mace_err) if n >= 20]
    ax.errorbar(
        plot_n, plot_y, yerr=plot_e,
        fmt="o-", color=C_MACE_FT, lw=2, ms=7, capsize=4,
        label="MACE-MP-0 fine-tuned", zorder=4,
    )
    # Dashed segment from clipped N=10 to N=20
    ax.plot([10, 20], [Y_CLIP - 0.01, plot_y[0]],
            color=C_MACE_FT, lw=1.5, ls="--", zorder=3)
    # N=10 marker at clip
    ax.scatter([10], [Y_CLIP - 0.01], color=C_MACE_FT, s=55, zorder=5, marker="o")

    # ── Fine-tuned CHGNet ─────────────────────────────────────────────────────
    if chgnet_lc:
        chg_raw = chgnet_lc.get("learning_curve", {})
        chg_n, chg_y, chg_err = [], [], []
        for n_str in sorted(chg_raw.keys(), key=int):
            n = int(n_str)
            v = chg_raw[n_str]
            chg_n.append(n)
            chg_y.append(v.get("mae_mean", np.nan))
            chg_err.append(v.get("mae_std", 0.0))
            ax.annotate(
                f"N={n}",
                xy=(n, v.get("mae_mean", 0)),
                xytext=(3, -11), textcoords="offset points",
                fontsize=7.5, color=C_CHG_FT,
            )
        ax.errorbar(
            chg_n, chg_y, yerr=chg_err,
            fmt="s-", color=C_CHG_FT, lw=2, ms=7, capsize=4,
            label="CHGNet fine-tuned", zorder=4,
        )

    # ── Crossover annotation: FT MACE reaches TensorNet accuracy ─────────────
    tn_mae = zs_mae.get("TensorNet", None)
    if tn_mae and len(plot_n) >= 2:
        # Find where MACE FT line crosses below TensorNet ZS (between N=60 and N=80)
        for i in range(len(plot_n) - 1):
            if plot_y[i] >= tn_mae >= plot_y[i + 1]:
                mid_n = (plot_n[i] + plot_n[i + 1]) / 2
                ax.annotate(
                    "Fine-tuned MACE\nmatches TensorNet\nzero-shot accuracy",
                    xy=(plot_n[i + 1], plot_y[i + 1]),
                    xytext=(plot_n[i + 1] - 22, plot_y[i + 1] + 0.06),
                    fontsize=8, color=C_MACE_FT,
                    arrowprops=dict(arrowstyle="->", color=C_MACE_FT, lw=0.9),
                )
                break

    # ── Secondary x-axis: approximate DFT hours ──────────────────────────────
    ax2 = ax.twiny()
    dft_hours = [compute_dft_budget_hr(n, train_pool, gpaw_times) for n in TRAIN_SIZES]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(TRAIN_SIZES)
    ax2.set_xticklabels([f"≈{h:.0f} h" for h in dft_hours], fontsize=8, color="#555")
    ax2.set_xlabel("Approximate DFT wall-clock budget", fontsize=9, color="#555", labelpad=6)
    ax2.tick_params(colors="#555")

    # ── Axes and labels ───────────────────────────────────────────────────────
    ax.set_xlabel("N DFT training points", fontsize=12)
    ax.set_ylabel(r"MAE on ΔG$_{H^*}$ (eV)", fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, Y_CLIP)
    ax.set_xticks(TRAIN_SIZES)

    # Legend (fine-tuned curves only; zero-shot are labelled inline)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(True, axis="both", alpha=0.2, ls="--")

    ax.set_title(
        r"Fine-tuning learning curves vs zero-shot baselines",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    VIZ_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 14 — Cost / Accuracy Trade-off Figure (v2)")
    print("=" * 65)

    # 1. Zero-shot baselines
    print("\n[1] Loading zero-shot baselines...")
    zero_shot = {}
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if summary_path.exists():
        bsum = json.load(open(summary_path))
        for model, mdata in bsum.get("by_model", {}).items():
            if isinstance(mdata, dict) and "mae_eV" in mdata:
                zero_shot[model] = mdata
        print(f"  {list(zero_shot.keys())}")
    else:
        print("  [warn] benchmark_summary.json not found")

    # 2. MACE learning curve
    print("\n[2] Loading MACE learning curve...")
    mace_lc = {}
    lc_path = RESULTS_DIR / "learning_curves.json"
    if lc_path.exists():
        raw = json.load(open(lc_path))
        for k, v in raw.get("learning_curve", {}).items():
            mace_lc[int(k)] = v
        print(f"  MACE LC: N={sorted(mace_lc.keys())}")
    else:
        print("  [warn] learning_curves.json not found — run script 09 first")

    # 3. CHGNet learning curve
    print("\n[3] Loading CHGNet learning curve...")
    chgnet_lc = None
    chgnet_lc_path = RESULTS_DIR / "chgnet_lc.json"
    if chgnet_lc_path.exists():
        chgnet_lc = json.load(open(chgnet_lc_path))
        print(f"  CHGNet LC: N={sorted(chgnet_lc.get('learning_curve', {}).keys(), key=int)}")
    else:
        print("  [warn] chgnet_lc.json not found — run script 13 first")

    # 4. Timing
    print("\n[4] Computing timings...")
    gpaw_times = gpaw_time_per_site()
    mace_times = mace_ft_times_by_n()
    print(f"  GPAW: " + ", ".join(f"Cu{sz}={gpaw_times.get(sz,0)/60:.1f}min" for sz in [10,20,30,40,50]))
    print(f"  MACE FT: " + ", ".join(f"N={n}={t[0]/60:.1f}min" for n,t in sorted(mace_times.items())))

    # 5. Train pool + DFT budget table
    print("\n[5] DFT budget breakdown:")
    train_pool = build_train_pool()
    print(f"  Train pool: {len(train_pool)} sites")
    print(f"  {'N':>4}  {'DFT (hr)':>9}  {'MACE FT (min)':>14}  {'Total (hr)':>11}")
    print("  " + "-" * 44)
    for n in TRAIN_SIZES:
        dft_hr = compute_dft_budget_hr(n, train_pool, gpaw_times)
        ft_s   = mace_times.get(n, (0, 0))[0]
        print(f"  {n:>4}  {dft_hr:>9.1f}  {ft_s/60:>14.1f}  {dft_hr + ft_s/3600:>11.1f}")

    # 6. Generate figure
    print("\n[6] Generating Fig 11 (v2)...")
    fig11_path = VIZ_DIR / "fig11_cost_accuracy.png"
    plot_cost_accuracy_v2(
        mace_lc    = mace_lc,
        chgnet_lc  = chgnet_lc,
        zero_shot  = zero_shot,
        mace_times = mace_times,
        gpaw_times = gpaw_times,
        train_pool = train_pool,
        output_path = fig11_path,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
