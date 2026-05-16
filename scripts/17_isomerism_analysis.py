"""
17_isomerism_analysis.py

Isomerism Analysis — How cluster geometry (isomer) affects ΔGH* and MLIP errors
---------------------------------------------------------------------------------
Quantifies geometry-induced spread in DFT ΔGH* across the 3 isomers (_00/_01/_02)
for each cluster size, and tests whether MLIP prediction errors are consistent
across isomers.

Outputs:
  viz/fig_isomerism.png           — two-panel figure
  results/isomerism_results.json  — per-geometry statistics

Usage:
    python3 scripts/17_isomerism_analysis.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
GPAW_DIR    = RESULTS_DIR / "gpaw"
VIZ_DIR     = PROJECT_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)

SIZES    = [10, 20, 30, 40, 50]
GEOMS    = ["00", "01", "02"]

# ── Color scheme ──────────────────────────────────────────────────────────────
COLORS = {
    "TensorNet":  "#2196F3",   # blue
    "MACE-MP-0":  "#FF6B35",   # orange
    "CHGNet":     "#4CAF50",   # green
    "DFT/GPAW":   "#1a1a2e",   # navy
}
GEOM_MARKERS = ["o", "s", "^"]   # _00, _01, _02
GEOM_COLORS  = ["#1a1a2e", "#555577", "#8888aa"]  # dark → light navy shades

# Use seaborn-v0_8-whitegrid if available, else whitegrid
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.linewidth":    1.2,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.framealpha": 0.85,
    "figure.dpi":        150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gpaw_all():
    """Return {config_id: [site_dict, ...]} for all GPAW results."""
    data = {}
    for sz in SIZES:
        for geom in GEOMS:
            rf = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_{geom}.json"
            if not rf.exists():
                continue
            d = json.load(open(rf))
            config_id = d.get("config_id", f"Cu{sz}_{geom}")
            ok_sites = [s for s in d["sites"] if s.get("status") == "ok"]
            data[config_id] = ok_sites
    return data


def load_mlip_all():
    """
    Return per-model dict: {model_name: {config_id: {site_global_id: dgh_eV}}}
    Models: MACE-MP-0, CHGNet, TensorNet
    """
    models = {}

    # MACE-MP-0 — dgh_mace_eV lives in GPAW files
    mace = {}
    for sz in SIZES:
        for geom in GEOMS:
            rf = GPAW_DIR / f"Cu{sz}" / f"results_Cu{sz}_{geom}.json"
            if not rf.exists():
                continue
            d = json.load(open(rf))
            config_id = d.get("config_id", f"Cu{sz}_{geom}")
            mace[config_id] = {
                s["site_global_id"]: s["dgh_mace_eV"]
                for s in d["sites"]
                if s.get("status") == "ok" and s.get("dgh_mace_eV") is not None
            }
    models["MACE-MP-0"] = mace

    # CHGNet
    chgnet_raw = json.load(open(RESULTS_DIR / "chgnet" / "results_chgnet.json"))
    chgnet = {}
    for s in chgnet_raw["sites"]:
        if s.get("status") == "ok" and s.get("dgh_chgnet_eV") is not None:
            chgnet.setdefault(s["config_id"], {})[s["site_global_id"]] = s["dgh_chgnet_eV"]
    models["CHGNet"] = chgnet

    # TensorNet
    tn_raw = json.load(open(RESULTS_DIR / "tensornet" / "results_tensornet.json"))
    tn = {}
    for s in tn_raw["sites"]:
        if s.get("status") == "ok" and s.get("dgh_tensornet_eV") is not None:
            tn.setdefault(s["config_id"], {})[s["site_global_id"]] = s["dgh_tensornet_eV"]
    models["TensorNet"] = tn

    return models


# ── Statistics ────────────────────────────────────────────────────────────────

def per_geometry_stats(gpaw_data, mlip_models):
    """
    For each (size, geom) combination compute:
      - DFT: mean ΔGH*, std across sites
      - Per model: MAE vs GPAW, Spearman ρ vs GPAW

    Returns nested dict: results[sz][geom] = { ... }
    """
    results = {}
    for sz in SIZES:
        results[sz] = {}
        for geom in GEOMS:
            config_id = f"Cu{sz}_{geom}"
            sites = gpaw_data.get(config_id, [])
            if not sites:
                results[sz][geom] = None
                continue

            dgh_gpaw = np.array([s["dgh_gpaw_eV"] for s in sites])
            entry = {
                "config_id":    config_id,
                "n_sites":      len(dgh_gpaw),
                "dft_mean":     float(np.mean(dgh_gpaw)),
                "dft_std":      float(np.std(dgh_gpaw)),
                "dft_min":      float(np.min(dgh_gpaw)),
                "dft_max":      float(np.max(dgh_gpaw)),
                "models":       {},
            }

            for model_name, model_data in mlip_models.items():
                cfg_dict = model_data.get(config_id, {})
                # Match by site_global_id
                pairs = []
                for s in sites:
                    gid = s["site_global_id"]
                    if gid in cfg_dict:
                        pairs.append((s["dgh_gpaw_eV"], cfg_dict[gid]))

                if len(pairs) < 2:
                    entry["models"][model_name] = None
                    continue

                gpaw_arr = np.array([p[0] for p in pairs])
                pred_arr = np.array([p[1] for p in pairs])
                mae  = float(np.mean(np.abs(gpaw_arr - pred_arr)))
                rmse = float(np.sqrt(np.mean((gpaw_arr - pred_arr) ** 2)))
                rho  = float(stats.spearmanr(gpaw_arr, pred_arr).statistic) if len(pairs) >= 3 else np.nan

                entry["models"][model_name] = {
                    "n_matched": len(pairs),
                    "mae_eV":    mae,
                    "rmse_eV":   rmse,
                    "spearman_rho": rho,
                }

            results[sz][geom] = entry

    return results


# ── Summary statistics ─────────────────────────────────────────────────────────

def compute_summary(results, mlip_models):
    """Return headline stats: max geometry spread and MLIP MAE variance."""
    max_spread   = 0.0
    max_spread_label = ""
    model_mae_variance = {}

    for sz in SIZES:
        means = []
        for geom in GEOMS:
            e = results[sz].get(geom)
            if e:
                means.append(e["dft_mean"])
        if len(means) >= 2:
            spread = max(means) - min(means)
            if spread > max_spread:
                max_spread = spread
                max_spread_label = f"Cu{sz}"

    for model_name in mlip_models:
        maes_all = []
        for sz in SIZES:
            sz_maes = []
            for geom in GEOMS:
                e = results[sz].get(geom)
                if e and e["models"].get(model_name):
                    sz_maes.append(e["models"][model_name]["mae_eV"])
            if len(sz_maes) >= 2:
                maes_all.append(float(np.std(sz_maes)))
        model_mae_variance[model_name] = float(np.mean(maes_all)) if maes_all else np.nan

    return {
        "max_geom_spread_eV":   round(max_spread, 3),
        "max_geom_spread_at":   max_spread_label,
        "model_mae_geom_std":   {k: round(v, 3) if not np.isnan(v) else None
                                 for k, v in model_mae_variance.items()},
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(results, summary, mlip_models):
    """
    Two-panel figure:
      Left:  DFT ΔGH* distribution per geometry per size (strip/scatter)
      Right: Per-geometry MAE for each model (grouped bars)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)

    x_positions = {sz: i for i, sz in enumerate(SIZES)}
    x_labels    = [f"Cu$_{{{sz}}}$" for sz in SIZES]
    n_geoms     = len(GEOMS)

    # ── Left panel — DFT ΔGH* distribution ────────────────────────────────────
    ax1.set_title("DFT ΔG$_{H*}$ Spread Across Isomers", fontweight="bold")
    ax1.set_xlabel("Cluster size")
    ax1.set_ylabel("ΔG$_{H*}$ (eV)")

    jitter_range = 0.12
    offsets = np.linspace(-jitter_range, jitter_range, n_geoms)

    for gi, geom in enumerate(GEOMS):
        label = f"Isomer _{geom}"
        for sz in SIZES:
            xi = x_positions[sz]
            e = results[sz].get(geom)
            if not e:
                continue
            sites_for_geom = []
            # Collect individual site DGH values
            config_id = f"Cu{sz}_{geom}"
            # Already have mean/std but we need all values for strip plot
            # Re-use the stats entry — we'll plot mean ± std as error bars
            # plus a scatter of the mean point
            y     = e["dft_mean"]
            yerr  = e["dft_std"]
            x_off = xi + offsets[gi]
            ax1.errorbar(
                x_off, y, yerr=yerr,
                fmt=GEOM_MARKERS[gi],
                color=GEOM_COLORS[gi],
                markersize=7,
                capsize=4,
                elinewidth=1.5,
                label=label if sz == SIZES[0] else "_nolegend_",
            )

    # Draw a horizontal line at ΔGH*=0 (thermoneutral)
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Thermoneutral")
    ax1.set_xticks(list(x_positions.values()))
    ax1.set_xticklabels(x_labels)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(-0.6, len(SIZES) - 0.4)

    # ── Right panel — Per-geometry MAE grouped bars ────────────────────────────
    ax2.set_title("Per-Isomer MAE for Each Model", fontweight="bold")
    ax2.set_xlabel("Cluster size")
    ax2.set_ylabel("MAE (eV)")

    model_names = list(mlip_models.keys())
    n_models    = len(model_names)
    bar_width   = 0.22
    geom_alphas = [1.0, 0.65, 0.35]   # _00 full, _01 medium, _02 faint
    geom_hatches= ["", "//", ".."]

    # Group layout: for each size, we have n_models clusters, each with 3 bars
    group_width  = n_models * bar_width + 0.1
    group_starts = {sz: i * group_width for i, sz in enumerate(SIZES)}

    legend_patches = []
    for mi, model in enumerate(model_names):
        color = COLORS[model]
        model_patch = mpatches.Patch(color=color, label=model)
        legend_patches.append(model_patch)
        for gi, geom in enumerate(GEOMS):
            for sz in SIZES:
                e = results[sz].get(geom)
                if not e or not e["models"].get(model):
                    continue
                mae = e["models"][model]["mae_eV"]
                x_pos = group_starts[sz] + mi * bar_width + gi * (bar_width / n_geoms)
                ax2.bar(
                    x_pos, mae,
                    width=bar_width / n_geoms,
                    color=color,
                    alpha=geom_alphas[gi],
                    hatch=geom_hatches[gi],
                    edgecolor="white",
                    linewidth=0.5,
                )

    # X-tick at center of each group
    tick_xs = [group_starts[sz] + (n_models * bar_width) / 2 for sz in SIZES]
    ax2.set_xticks(tick_xs)
    ax2.set_xticklabels(x_labels)

    # Geom legend (hatch patches)
    geom_patches = [
        mpatches.Patch(facecolor="gray", alpha=geom_alphas[gi],
                       hatch=geom_hatches[gi], label=f"Isomer _{GEOMS[gi]}")
        for gi in range(n_geoms)
    ]
    leg1 = ax2.legend(handles=legend_patches, fontsize=8, loc="upper left",
                      title="Model", title_fontsize=8)
    ax2.add_artist(leg1)
    ax2.legend(handles=geom_patches, fontsize=8, loc="upper right",
               title="Geom.", title_fontsize=8)

    # ── Stats text box ─────────────────────────────────────────────────────────
    mae_var_str = ", ".join(
        f"{m}: ±{v:.3f} eV" if v is not None else f"{m}: N/A"
        for m, v in summary["model_mae_geom_std"].items()
    )
    stats_text = (
        f"Max geometry-induced ΔG$_{{H*}}$ spread: "
        f"{summary['max_geom_spread_eV']:.3f} eV at {summary['max_geom_spread_at']}\n"
        f"MLIP MAE isomer variance (σ): {mae_var_str}"
    )
    fig.text(
        0.5, 0.01, stats_text,
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.85),
    )

    fig.suptitle(
        "Cluster Isomerism: Geometry Sensitivity of ΔG$_{H*}$ and MLIP Errors",
        fontsize=12, fontweight="bold", y=1.02,
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading GPAW data ...")
    gpaw_data   = load_gpaw_all()
    print(f"  Loaded {sum(len(v) for v in gpaw_data.values())} sites across "
          f"{len(gpaw_data)} configurations")

    print("Loading MLIP predictions ...")
    mlip_models = load_mlip_all()
    for name, cfg_dict in mlip_models.items():
        n = sum(len(v) for v in cfg_dict.values())
        print(f"  {name}: {n} predictions")

    print("Computing per-geometry statistics ...")
    results = per_geometry_stats(gpaw_data, mlip_models)

    print("Computing summary ...")
    summary = compute_summary(results, mlip_models)
    print(f"  Max geometry spread: {summary['max_geom_spread_eV']:.3f} eV "
          f"at {summary['max_geom_spread_at']}")
    print(f"  MLIP MAE isomer σ: {summary['model_mae_geom_std']}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "isomerism_results.json"
    output = {
        "summary":    summary,
        "per_size_geom": {
            f"Cu{sz}_{geom}": results[sz][geom]
            for sz in SIZES for geom in GEOMS
            if results[sz].get(geom) is not None
        },
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_json}")

    # ── Save figure ───────────────────────────────────────────────────────────
    print("Generating figure ...")
    fig = make_figure(results, summary, mlip_models)
    out_png = VIZ_DIR / "fig_isomerism.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # ── Per-size table ─────────────────────────────────────────────────────────
    print("\nPer-geometry MAE table (eV):")
    header = f"{'Config':<12}" + "".join(f"  {m:<12}" for m in mlip_models)
    print(header)
    for sz in SIZES:
        for geom in GEOMS:
            e = results[sz].get(geom)
            if not e:
                continue
            row = f"Cu{sz}_{geom}    "
            for model in mlip_models:
                ms = e["models"].get(model)
                row += f"  {ms['mae_eV']:.4f} eV    " if ms else "  N/A           "
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
