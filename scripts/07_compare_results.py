"""
07_compare_results.py

MLIP vs GPAW Benchmark — Analysis & Figures
---------------------------------------------
Compares MACE-MP-0, CHGNet, and TensorNet against GPAW PBE
for ΔGH* prediction on Cu nanoclusters (Cu10–Cu50).

Produces:
  viz/fig1_correlation_all.png     — 2×2 scatter: MACE/CHGNet/TensorNet vs GPAW
  viz/fig2_mae_vs_size.png         — MAE & RMSE vs cluster size (line plot)
  viz/fig3_mae_by_sitetype.png     — MAE by site type (atop/bridge/hollow)
  viz/fig4_dgh_distributions.png   — ΔGH* histograms: GPAW vs each MLIP
  results/benchmark_summary.json   — full MAE/RMSE table (machine-readable)

Runs on partial data — sizes with no GPAW results yet are skipped.
Re-run after Cu50 finishes to get complete figures.

Usage:
    python3 07_compare_results.py
    python3 07_compare_results.py --no-show   # save only, don't open windows
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).parent.parent
RESULTS  = PROJECT / "results"
GPAW_DIR = RESULTS / "gpaw"
VIZ_DIR  = PROJECT / "viz"
VIZ_DIR.mkdir(exist_ok=True)

SIZES = [10, 20, 30, 40, 50]

# Publication-quality style
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.size':       11,
    'axes.linewidth':  1.2,
    'axes.labelsize':  12,
    'axes.titlesize':  12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.framealpha': 0.85,
    'figure.dpi':      150,
})

COLORS = {
    'MACE':      '#2196F3',   # blue
    'CHGNet':    '#F44336',   # red
    'TensorNet': '#4CAF50',   # green
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-show', action='store_true',
                   help='Save figures without displaying')
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_gpaw_sites() -> dict:
    """Load all completed GPAW results (all cluster IDs). Returns {site_global_id: dict}."""
    sites = {}
    for sz in SIZES:
        sz_total = 0
        for cid_suffix in ['00', '01', '02']:
            cid = f'Cu{sz}_{cid_suffix}'
            result_f = GPAW_DIR / f'Cu{sz}' / f'results_{cid}.json'
            ckpt_f   = GPAW_DIR / f'Cu{sz}' / f'checkpoint_{cid}.json'

            data = None
            if result_f.exists():
                with open(result_f) as f:
                    data = json.load(f)
                src = 'final'
            elif ckpt_f.exists():
                with open(ckpt_f) as f:
                    ckpt = json.load(f)
                data = {'sites': ckpt['site_results']}
                src = 'checkpoint'

            if data:
                good = [s for s in data['sites'] if s.get('dgh_gpaw_eV') is not None]
                for s in good:
                    sites[s['site_global_id']] = s
                sz_total += len(good)

        if sz_total:
            print(f"  Cu{sz}: {sz_total} GPAW sites loaded")
        else:
            print(f"  Cu{sz}: no GPAW data yet — skipping")
    return sites


def load_mlip_sites(fname: str, dgh_key: str) -> dict:
    """Load MLIP results. Returns {site_global_id: dgh_value}."""
    with open(RESULTS / fname) as f:
        data = json.load(f)
    return {s['site_global_id']: s[dgh_key]
            for s in data['sites'] if s.get(dgh_key) is not None}


def build_paired(gpaw: dict, mlip: dict) -> tuple:
    """
    Returns (gpaw_vals, mlip_vals, sizes, types) arrays for sites
    present in both GPAW and MLIP results.
    """
    gpaw_vals, mlip_vals, sizes, types = [], [], [], []
    for sid, g in gpaw.items():
        if sid in mlip:
            gpaw_vals.append(g['dgh_gpaw_eV'])
            mlip_vals.append(mlip[sid])
            sizes.append(g['size'])
            types.append(g['site_type'])
    return (np.array(gpaw_vals), np.array(mlip_vals),
            np.array(sizes), np.array(types))


def metrics(gpaw_vals, mlip_vals):
    delta = mlip_vals - gpaw_vals
    mae   = float(np.mean(np.abs(delta)))
    rmse  = float(np.sqrt(np.mean(delta**2)))
    # Pearson r
    if len(gpaw_vals) > 1:
        r = float(np.corrcoef(gpaw_vals, mlip_vals)[0, 1])
    else:
        r = float('nan')
    return mae, rmse, r


# ── Figure 1: Correlation scatter (3-panel) ──────────────────────────────────

def fig_correlation(gpaw, mace_d, chgnet_d, tensornet_d):
    models = [
        ('MACE-MP-0',  mace_d,      'dgh_mace_eV',  COLORS['MACE']),
        ('CHGNet',     chgnet_d,    None,            COLORS['CHGNet']),
        ('TensorNet',  tensornet_d, None,            COLORS['TensorNet']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, (name, mlip_dict, _, color) in zip(axes, models):
        g_vals, m_vals, szs, _ = build_paired(gpaw, mlip_dict)
        if len(g_vals) == 0:
            ax.set_title(f'{name}\n(no data)')
            continue

        mae, rmse, r = metrics(g_vals, m_vals)

        # Scatter colored by size
        size_vals = sorted(set(szs))
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(size_vals), max(size_vals))

        for sz in size_vals:
            mask = szs == sz
            ax.scatter(g_vals[mask], m_vals[mask],
                       c=[cmap(norm(sz))]*mask.sum(),
                       s=35, alpha=0.7, edgecolors='none',
                       label=f'Cu{sz}')

        # Diagonal
        lims = [min(g_vals.min(), m_vals.min()) - 0.05,
                max(g_vals.max(), m_vals.max()) + 0.05]
        ax.plot(lims, lims, 'k--', lw=1.0, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)

        ax.set_xlabel('ΔGH* GPAW PBE (eV)')
        ax.set_ylabel(f'ΔGH* {name} (eV)')
        ax.set_title(f'{name}', fontweight='bold')

        textstr = f'MAE  = {mae:.3f} eV\nRMSE = {rmse:.3f} eV\nr = {r:.3f}\nn = {len(g_vals)}'
        ax.text(0.04, 0.96, textstr, transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        if sz == size_vals[-1]:
            ax.legend(title='Size', fontsize=8, title_fontsize=8,
                      loc='lower right', markerscale=1.2)

    fig.suptitle('MLIP vs GPAW PBE: ΔGH* for Cu Nanoclusters',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = VIZ_DIR / 'fig1_correlation_all.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    return fig


# ── Figure 2: MAE & RMSE vs size ─────────────────────────────────────────────

def fig_mae_vs_size(gpaw, mace_d, chgnet_d, tensornet_d):
    model_data = [
        ('MACE-MP-0',  mace_d,      COLORS['MACE'],      'o-'),
        ('CHGNet',     chgnet_d,    COLORS['CHGNet'],    's-'),
        ('TensorNet',  tensornet_d, COLORS['TensorNet'], '^-'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for name, mlip_dict, color, marker in model_data:
        mae_by_sz, rmse_by_sz, sz_list = [], [], []
        for sz in SIZES:
            g_vals, m_vals, _, _ = build_paired(
                {k: v for k, v in gpaw.items() if v['size'] == sz},
                mlip_dict
            )
            if len(g_vals) < 2:
                continue
            mae, rmse, _ = metrics(g_vals, m_vals)
            mae_by_sz.append(mae)
            rmse_by_sz.append(rmse)
            sz_list.append(sz)

        if not sz_list:
            continue
        ax1.plot(sz_list, mae_by_sz,  marker, color=color, lw=2,
                 markersize=7, label=name)
        ax2.plot(sz_list, rmse_by_sz, marker, color=color, lw=2,
                 markersize=7, label=name)

    for ax, ylabel, title in [
        (ax1, 'MAE (eV)',  'Mean Absolute Error vs Cluster Size'),
        (ax2, 'RMSE (eV)', 'RMSE vs Cluster Size'),
    ]:
        ax.set_xlabel('Cluster size (N atoms)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([f'Cu{s}' for s in SIZES])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.1, color='gray', lw=0.8, ls=':', alpha=0.6,
                   label='0.1 eV target')

    fig.suptitle('MLIP Accuracy vs Cluster Size', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    out = VIZ_DIR / 'fig2_mae_vs_size.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    return fig


# ── Figure 3: MAE by site type ────────────────────────────────────────────────

def fig_mae_by_sitetype(gpaw, mace_d, chgnet_d, tensornet_d):
    site_types = ['atop', 'bridge', 'hollow']
    model_data = [
        ('MACE-MP-0',  mace_d,      COLORS['MACE']),
        ('CHGNet',     chgnet_d,    COLORS['CHGNet']),
        ('TensorNet',  tensornet_d, COLORS['TensorNet']),
    ]

    x = np.arange(len(site_types))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (name, mlip_dict, color) in enumerate(model_data):
        mae_by_type = []
        counts = []
        for stype in site_types:
            g_vals, m_vals, _, _ = build_paired(
                {k: v for k, v in gpaw.items() if v['site_type'] == stype},
                mlip_dict
            )
            if len(g_vals) > 0:
                mae, _, _ = metrics(g_vals, m_vals)
                mae_by_type.append(mae)
                counts.append(len(g_vals))
            else:
                mae_by_type.append(0)
                counts.append(0)

        bars = ax.bar(x + i * width, mae_by_type, width,
                      label=name, color=color, alpha=0.85, edgecolor='white')

        # Count labels above bars
        for bar, n in zip(bars, counts):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'n={n}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xlabel('Site type')
    ax.set_ylabel('MAE vs GPAW (eV)')
    ax.set_title('MLIP Accuracy by Adsorption Site Type', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in site_types])
    ax.axhline(0.1, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    out = VIZ_DIR / 'fig3_mae_by_sitetype.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    return fig


# ── Figure 4: ΔGH* distributions ─────────────────────────────────────────────

def fig_dgh_distributions(gpaw, mace_d, chgnet_d, tensornet_d):
    model_data = [
        ('MACE-MP-0',  mace_d,      COLORS['MACE']),
        ('CHGNet',     chgnet_d,    COLORS['CHGNet']),
        ('TensorNet',  tensornet_d, COLORS['TensorNet']),
    ]

    # Collect all GPAW values + corresponding MLIP values
    gpaw_all = np.array([v['dgh_gpaw_eV'] for v in gpaw.values()])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    bins = np.linspace(-0.6, 0.7, 26)

    for ax, (name, mlip_dict, color) in zip(axes, model_data):
        g_vals, m_vals, _, _ = build_paired(gpaw, mlip_dict)
        if len(g_vals) == 0:
            continue

        ax.hist(g_vals, bins=bins, alpha=0.5, color='#555555',
                label='GPAW', edgecolor='white', lw=0.5)
        ax.hist(m_vals, bins=bins, alpha=0.6, color=color,
                label=name, edgecolor='white', lw=0.5)

        ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.5,
                   label='Thermoneutral')
        ax.set_xlabel('ΔGH* (eV)')
        ax.set_title(f'GPAW vs {name}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel('Count')
    fig.suptitle('ΔGH* Distributions: GPAW vs MLIPs', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    out = VIZ_DIR / 'fig4_dgh_distributions.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(gpaw, mace_d, chgnet_d, tensornet_d) -> dict:
    model_data = [
        ('MACE-MP-0',  mace_d),
        ('CHGNet',     chgnet_d),
        ('TensorNet',  tensornet_d),
    ]

    summary = {'by_model': {}, 'by_size': {}}

    for name, mlip_dict in model_data:
        g_all, m_all, _, _ = build_paired(gpaw, mlip_dict)
        if len(g_all) == 0:
            continue
        mae, rmse, r = metrics(g_all, m_all)
        summary['by_model'][name] = {
            'n': len(g_all),
            'mae_eV': round(mae, 4),
            'rmse_eV': round(rmse, 4),
            'pearson_r': round(r, 4),
        }

        by_sz = {}
        for sz in SIZES:
            g_sz, m_sz, _, _ = build_paired(
                {k: v for k, v in gpaw.items() if v['size'] == sz},
                mlip_dict
            )
            if len(g_sz) < 2:
                continue
            mae_sz, rmse_sz, r_sz = metrics(g_sz, m_sz)
            by_sz[sz] = {
                'n': len(g_sz),
                'mae_eV': round(mae_sz, 4),
                'rmse_eV': round(rmse_sz, 4),
                'pearson_r': round(r_sz, 4),
            }
        summary['by_size'][name] = by_sz

    out = RESULTS / 'benchmark_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    return summary


def print_table(summary: dict):
    print(f"\n{'='*65}")
    print("MLIP vs GPAW PBE — Benchmark Summary")
    print(f"{'='*65}")

    # Overall
    print(f"\n{'Model':<14}  {'N':>5}  {'MAE (eV)':>10}  {'RMSE (eV)':>11}  {'Pearson r':>10}")
    print("  " + "-" * 52)
    for name, s in summary['by_model'].items():
        print(f"  {name:<12}  {s['n']:>5}  {s['mae_eV']:>10.4f}  "
              f"{s['rmse_eV']:>11.4f}  {s['pearson_r']:>10.4f}")

    # By size
    print(f"\n  MAE (eV) by cluster size:")
    print(f"  {'Model':<14}", end='')
    for sz in SIZES:
        print(f"  {'Cu'+str(sz):>7}", end='')
    print()
    print("  " + "-" * (14 + 9*len(SIZES)))
    for name in summary['by_size']:
        print(f"  {name:<14}", end='')
        for sz in SIZES:
            s = summary['by_size'][name].get(sz)
            val = f"{s['mae_eV']:.4f}" if s else '  —   '
            print(f"  {val:>7}", end='')
        print()
    print(f"{'='*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 65)
    print("07 — MLIP vs GPAW Benchmark Analysis")
    print("=" * 65)

    # Load all data
    print("\nLoading GPAW results...")
    gpaw = load_gpaw_sites()
    if not gpaw:
        print("No GPAW data found — run 04a/04b first.")
        return

    print("\nLoading MLIP results...")

    # MACE: dgh_mace_eV is stored in every GPAW site record
    mace_d = {sid: s['dgh_mace_eV'] for sid, s in gpaw.items()}

    chgnet_d    = load_mlip_sites('chgnet/results_chgnet.json',    'dgh_chgnet_eV')
    tensornet_d = load_mlip_sites('tensornet/results_tensornet.json', 'dgh_tensornet_eV')

    print(f"\n  GPAW:      {len(gpaw)} sites")
    print(f"  MACE:      {len(mace_d)} sites (paired)")
    print(f"  CHGNet:    {len(chgnet_d)} sites total in file")
    print(f"  TensorNet: {len(tensornet_d)} sites total in file")

    # Summary table
    print("\nBuilding summary...")
    summary = build_summary(gpaw, mace_d, chgnet_d, tensornet_d)
    print_table(summary)

    # Figures
    print("Generating figures...")
    fig_correlation(gpaw, mace_d, chgnet_d, tensornet_d)
    fig_mae_vs_size(gpaw, mace_d, chgnet_d, tensornet_d)
    fig_mae_by_sitetype(gpaw, mace_d, chgnet_d, tensornet_d)
    fig_dgh_distributions(gpaw, mace_d, chgnet_d, tensornet_d)

    n_gpaw = len(gpaw)
    n_total = sum(1 for sz in SIZES for _ in range({10:6,20:27,30:18,40:5,50:61}[sz]))
    if n_gpaw < n_total:
        remaining = n_total - n_gpaw
        print(f"\n  Note: {remaining} GPAW sites still pending (Cu50 running).")
        print(f"  Re-run this script when Cu50 finishes for complete figures.")
    else:
        print(f"\n  All {n_gpaw} sites complete — figures are final.")

    print(f"\nOutputs in viz/:")
    for f in sorted(VIZ_DIR.glob('fig*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
