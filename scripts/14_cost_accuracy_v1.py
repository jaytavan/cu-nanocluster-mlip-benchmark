"""
Script 14 — Cost / Accuracy Trade-off Figure
=============================================
Generates fig11: MAE on ΔGH* vs total compute cost (wall-clock hours).

X-axis: DFT budget (wall-clock hours) to produce N training points
        + ML fine-tuning time (GPU, wall-clock)
Y-axis: MAE on ΔGH* (eV) on the 29-site hold-out test set

Points:
- Zero-shot models (x=0): MACE-MP-0, CHGNet, TensorNet
- MACE-MP-0 fine-tuned at N=10,20,40,60,80 (mean ± std across 3 seeds)
- CHGNet fine-tuned at N=10,20,40,60,80 (if chgnet_lc.json exists)

Key message: DFT time dominates fine-tuning time (~10× at all N).
The GPU fine-tuning overhead is negligible relative to DFT budget.

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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
GPAW_DIR    = RESULTS_DIR / "gpaw"
VIZ_DIR     = PROJECT_DIR / "viz"

TRAIN_SIZES = [10, 20, 40, 60, 80]
TEST_FRAC   = 0.20


# ── Timing extraction ──────────────────────────────────────────────────────────

def gpaw_time_per_site() -> Dict[int, float]:
    """
    Return mean GPAW wall-clock time per site (seconds) broken down by cluster size.
    Used to estimate DFT budget for N training sites.
    """
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


def compute_dft_budget_hr(n_train: int, train_pool_sites: list, gpaw_times: Dict[int, float]) -> float:
    """
    Estimate DFT wall-clock hours to produce n_train training points.
    Uses the size distribution of the actual training pool for weighted averaging.
    """
    from collections import Counter
    size_counts = Counter(s["size"] for s in train_pool_sites)
    total_pool  = sum(size_counts.values())

    # Weighted mean time per site (based on pool size distribution)
    weighted_s = sum(
        (count / total_pool) * gpaw_times.get(sz, np.mean(list(gpaw_times.values())))
        for sz, count in size_counts.items()
    )
    return n_train * weighted_s / 3600.0


def mace_ft_times_by_n() -> Dict[int, Tuple[float, float]]:
    """
    Parse MACE fine-tuning wall-clock times from per-run log files.
    Returns {n_train: (mean_s, std_s)}.
    """
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
        timestamps = [ts_pat.search(line) for line in txt.split("\n")]
        timestamps = [ts.group(1) for ts in timestamps if ts]
        if len(timestamps) < 2:
            continue
        elapsed = (
            datetime.strptime(timestamps[-1], fmt) -
            datetime.strptime(timestamps[0], fmt)
        ).total_seconds()
        by_n.setdefault(n, []).append(elapsed)

    return {
        n: (float(np.mean(times)), float(np.std(times)))
        for n, times in by_n.items()
    }


def chgnet_ft_times_by_n(chgnet_lc: Dict) -> Dict[int, Tuple[float, float]]:
    """
    Extract CHGNet fine-tuning times from chgnet_lc.json (stored by script 13).
    Returns {n_train: (mean_s, std_s)}.
    """
    times = {}
    for n_str, v in chgnet_lc.get("learning_curve", {}).items():
        n = int(n_str)
        mean_s = v.get("ft_time_s_mean")
        std_s  = v.get("ft_time_s_std", 0.0)
        if mean_s is not None:
            times[n] = (float(mean_s), float(std_s))
    return times


def zero_shot_times() -> Dict[str, float]:
    """Inference-only wall-clock seconds for zero-shot predictions (all 145 sites)."""
    times = {}
    chgnet_res = RESULTS_DIR / "chgnet" / "results_chgnet.json"
    if chgnet_res.exists():
        d = json.load(open(chgnet_res))
        if d.get("total_time_s"):
            times["CHGNet"] = d["total_time_s"]
    tn_res = RESULTS_DIR / "tensornet" / "results_tensornet.json"
    if tn_res.exists():
        d = json.load(open(tn_res))
        if d.get("total_time_s"):
            times["TensorNet"] = d["total_time_s"]
    # MACE timing: from sites_mace.json total or estimate
    mace_res = RESULTS_DIR / "sites_mace.json"
    if mace_res.exists():
        try:
            d = json.load(open(mace_res))
            if isinstance(d, list) and d[0].get("time_s"):
                times["MACE-MP-0"] = sum(s.get("time_s", 0) for s in d)
        except Exception:
            pass
    return times


# ── Build train pool (needed for DFT budget weighting) ────────────────────────

def build_train_pool():
    """Reproduce the same train pool as scripts 09/13 (seed=42, 20% test)."""
    all_sites = []
    for sz in [10, 20, 30, 40, 50]:
        rfile = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_00.json"
        if not rfile.exists():
            continue
        data = json.load(open(rfile))
        all_sites.extend([s for s in data["sites"] if s["status"] == "ok"])

    rng = np.random.default_rng(seed=42)
    by_size = {}
    for s in all_sites:
        by_size.setdefault(s["size"], []).append(s)

    train_pool = []
    for sz, sites in sorted(by_size.items()):
        n_test = max(1, round(len(sites) * TEST_FRAC))
        idx    = rng.permutation(len(sites))
        train_pool.extend([sites[i] for i in idx[n_test:]])

    return train_pool


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_cost_accuracy(
    mace_lc:      Dict,
    chgnet_lc:    Optional[Dict],
    zero_shot:    Dict,
    mace_times:   Dict,
    chgnet_times: Dict,
    gpaw_times:   Dict,
    train_pool:   list,
    output_path:  Path,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        r"Cost / Accuracy Trade-off: ΔG$_{H^*}$ MAE vs Compute Investment",
        fontsize=13, fontweight="bold",
    )

    # ── colours ──────────────────────────────────────────────────────────────
    C_MACE_FT  = "#e74c3c"
    C_CHG_FT   = "#27ae60"
    C_MACE_ZS  = "#3498db"
    C_CHG_ZS   = "#2ecc71"
    C_TN_ZS    = "#9b59b6"

    # ── zero-shot baseline markers (x ≈ 0; tiny x for log axis safety) ──────
    zs_x_hr = 1e-3
    zs_mae  = {m: v["mae"] for m, v in zero_shot.items() if "mae" in v}

    zs_colors = {"MACE-MP-0": C_MACE_ZS, "CHGNet": C_CHG_ZS, "TensorNet": C_TN_ZS}
    zs_markers = {"MACE-MP-0": "D", "CHGNet": "s", "TensorNet": "^"}
    for model, mae in sorted(zs_mae.items()):
        ax.scatter(
            zs_x_hr, mae,
            marker=zs_markers.get(model, "o"),
            color=zs_colors.get(model, "gray"),
            s=90, zorder=5,
            label=f"{model} zero-shot",
        )

    # ── fine-tuned MACE ───────────────────────────────────────────────────────
    if mace_lc:
        mace_xs, mace_ys, mace_errs = [], [], []
        mace_xs_gpu = []     # GPU-only FT cost (no DFT)
        for n in sorted(mace_lc.keys()):
            v = mace_lc[n]
            mae_mean = v.get("mae_mean", v.get("mae", np.nan))
            mae_std  = v.get("mae_std", 0.0)
            dft_hr   = compute_dft_budget_hr(n, train_pool, gpaw_times)
            ft_mean_s, _ = mace_times.get(n, (0.0, 0.0))
            total_hr = dft_hr + ft_mean_s / 3600.0
            mace_xs.append(total_hr)
            mace_ys.append(mae_mean)
            mace_errs.append(mae_std)
            mace_xs_gpu.append(ft_mean_s / 3600.0)
            # Annotate N value
            ax.annotate(
                f"N={n}",
                xy=(total_hr, mae_mean),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color=C_MACE_FT, alpha=0.85,
            )
        ax.errorbar(
            mace_xs, mace_ys, yerr=mace_errs,
            fmt="o-", color=C_MACE_FT, lw=2, ms=7, capsize=4,
            label="MACE-MP-0 fine-tuned", zorder=4,
        )

    # ── fine-tuned CHGNet ─────────────────────────────────────────────────────
    if chgnet_lc:
        chg_lc_raw = chgnet_lc.get("learning_curve", {})
        if chg_lc_raw:
            chg_xs, chg_ys, chg_errs = [], [], []
            for n_str in sorted(chg_lc_raw.keys(), key=int):
                n = int(n_str)
                v = chg_lc_raw[n_str]
                mae_mean = v.get("mae_mean", np.nan)
                mae_std  = v.get("mae_std", 0.0)
                dft_hr   = compute_dft_budget_hr(n, train_pool, gpaw_times)
                ft_mean_s, _ = chgnet_times.get(n, (0.0, 0.0))
                total_hr = dft_hr + ft_mean_s / 3600.0
                chg_xs.append(total_hr)
                chg_ys.append(mae_mean)
                chg_errs.append(mae_std)
                ax.annotate(
                    f"N={n}",
                    xy=(total_hr, mae_mean),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=7.5, color=C_CHG_FT, alpha=0.85,
                )
            ax.errorbar(
                chg_xs, chg_ys, yerr=chg_errs,
                fmt="s-", color=C_CHG_FT, lw=2, ms=7, capsize=4,
                label="CHGNet fine-tuned", zorder=4,
            )

    # ── DFT-budget reference annotation ──────────────────────────────────────
    # Show how much of x is DFT vs GPU for N=80 MACE
    if mace_lc and mace_times.get(80):
        dft_hr_80 = compute_dft_budget_hr(80, train_pool, gpaw_times)
        ft_hr_80  = mace_times[80][0] / 3600.0
        total_80  = dft_hr_80 + ft_hr_80
        ax.annotate(
            f"DFT={dft_hr_80:.0f} h\nGPU FT={ft_hr_80*60:.0f} min",
            xy=(total_80, mace_lc.get(80, {}).get("mae_mean", 0.09)),
            xytext=(-60, 20), textcoords="offset points",
            fontsize=7.5, color="#555",
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
        )

    ax.set_xlabel("Total wall-clock investment (DFT hours + GPU fine-tuning hours)", fontsize=11)
    ax.set_ylabel(r"MAE on ΔG$_{H^*}$ (eV)", fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, ls="--")

    # Secondary x-axis label note
    ax.text(
        0.02, 0.04,
        "Zero-shot models plotted at x=0.001 h (inference only, ~5 s)",
        transform=ax.transAxes, fontsize=7.5, color="#555", style="italic",
    )

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    VIZ_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("Script 14 — Cost / Accuracy Trade-off Figure")
    print("=" * 65)

    # ── 1. Load zero-shot baselines ─────────────────────────────────────────
    print("\n[1] Loading zero-shot baselines...")
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

    # ── 2. Load MACE LC results ─────────────────────────────────────────────
    print("\n[2] Loading MACE fine-tuning learning curves...")
    mace_lc = {}
    lc_path = RESULTS_DIR / "learning_curves.json"
    if lc_path.exists():
        raw = json.load(open(lc_path))
        for k, v in raw.get("learning_curve", {}).items():
            mace_lc[int(k)] = v
        print(f"  MACE LC: N={sorted(mace_lc.keys())}")
    else:
        print("  [warn] learning_curves.json not found — run script 09 first")

    # ── 3. Load CHGNet LC results (optional) ────────────────────────────────
    print("\n[3] Loading CHGNet fine-tuning learning curves...")
    chgnet_lc = None
    chgnet_lc_path = RESULTS_DIR / "chgnet_lc.json"
    if chgnet_lc_path.exists():
        chgnet_lc = json.load(open(chgnet_lc_path))
        n_chg = list(chgnet_lc.get("learning_curve", {}).keys())
        print(f"  CHGNet LC: N={sorted(n_chg, key=int)}")
    else:
        print("  [warn] chgnet_lc.json not found — CHGNet curve will be absent")
        print("         Run script 13 first, then re-run this script.")

    # ── 4. Extract timing ────────────────────────────────────────────────────
    print("\n[4] Extracting compute timings...")
    gpaw_times  = gpaw_time_per_site()
    print(f"  GPAW mean time/site: " +
          ", ".join(f"Cu{sz}={gpaw_times.get(sz, 0):.0f}s" for sz in [10,20,30,40,50]))

    mace_times  = mace_ft_times_by_n()
    print(f"  MACE FT times (mean): " +
          ", ".join(f"N={n}={v[0]:.0f}s" for n, v in sorted(mace_times.items())))

    chgnet_times = {}
    if chgnet_lc:
        chgnet_times = chgnet_ft_times_by_n(chgnet_lc)
        if chgnet_times:
            print(f"  CHGNet FT times (mean): " +
                  ", ".join(f"N={n}={v[0]:.0f}s" for n, v in sorted(chgnet_times.items())))
        else:
            print("  CHGNet FT times: not stored in chgnet_lc.json (run script 13 again)")

    # ── 5. Build train pool for DFT budget weighting ─────────────────────────
    print("\n[5] Building train pool for DFT budget estimation...")
    train_pool = build_train_pool()
    print(f"  Train pool: {len(train_pool)} sites")

    # Show DFT budget estimates
    print("\n  DFT budget breakdown:")
    print(f"  {'N':>5}  {'DFT (hr)':>10}  {'MACE FT (min)':>14}  {'Total MACE (hr)':>16}")
    print("  " + "-" * 52)
    for n in TRAIN_SIZES:
        dft_hr = compute_dft_budget_hr(n, train_pool, gpaw_times)
        ft_mean_s, _ = mace_times.get(n, (0.0, 0.0))
        total_hr = dft_hr + ft_mean_s / 3600.0
        print(f"  {n:>5}  {dft_hr:>10.2f}  {ft_mean_s/60:>14.1f}  {total_hr:>16.2f}")

    # ── 6. Generate figure ────────────────────────────────────────────────────
    print("\n[6] Generating Fig 11...")
    fig11_path = VIZ_DIR / "fig11_cost_accuracy.png"
    plot_cost_accuracy(
        mace_lc      = mace_lc,
        chgnet_lc    = chgnet_lc,
        zero_shot    = zero_shot,
        mace_times   = mace_times,
        chgnet_times = chgnet_times,
        gpaw_times   = gpaw_times,
        train_pool   = train_pool,
        output_path  = fig11_path,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COST / ACCURACY SUMMARY")
    print("=" * 65)
    print("\nZero-shot (no DFT budget):")
    for model, m in zero_shot.items():
        print(f"  {model:15s}: MAE={m.get('mae', '?'):.4f} eV")

    if mace_lc:
        print("\nMACE fine-tuned:")
        for n in sorted(mace_lc.keys()):
            v = mace_lc[n]
            dft_hr = compute_dft_budget_hr(n, train_pool, gpaw_times)
            ft_s, _ = mace_times.get(n, (0, 0))
            print(f"  N={n:3d}: MAE={v.get('mae_mean', 0):.3f}±{v.get('mae_std', 0):.3f} eV  "
                  f"DFT={dft_hr:.1f}h  FT={ft_s/60:.0f}min  total={dft_hr+ft_s/3600:.1f}h")

    if chgnet_lc and chgnet_lc.get("learning_curve"):
        print("\nCHGNet fine-tuned:")
        for n_str, v in sorted(chgnet_lc["learning_curve"].items(), key=lambda x: int(x[0])):
            n = int(n_str)
            dft_hr = compute_dft_budget_hr(n, train_pool, gpaw_times)
            ft_s, _ = chgnet_times.get(n, (0, 0))
            print(f"  N={n:3d}: MAE={v.get('mae_mean', 0):.3f}±{v.get('mae_std', 0):.3f} eV  "
                  f"DFT={dft_hr:.1f}h  FT={ft_s/60:.0f}min  total={dft_hr+ft_s/3600:.1f}h")

    print(f"\nFig 11: {fig11_path}")
    print("Done.")


if __name__ == "__main__":
    main()
