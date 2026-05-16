"""
08_publication_figures.py

Publication-Quality Supplementary Figures
------------------------------------------
Adds to viz/:
  fig5_che_free_energy.png   — CHE free energy diagrams (all sizes, all models)
  fig6_structure_gallery.png — Rendered cluster+H structures (best site per size)
  fig7_volcano.png           — ΔGH* distribution vs size (pseudo-volcano)

Figures 1–4 are updated in-place by re-running 07_compare_results.py.
This script only adds new figures — no duplicates.

Usage:
    python3 08_publication_figures.py
    python3 08_publication_figures.py --sizes 10 20 30 40   # skip Cu50 if incomplete
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).parent.parent
RESULTS  = PROJECT / "results"
GPAW_DIR = RESULTS / "gpaw"
VIZ_DIR  = PROJECT / "viz"
VIZ_DIR.mkdir(exist_ok=True)

SIZES = [10, 20, 30, 40, 50]

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.linewidth':   1.2,
    'axes.labelsize':   12,
    'axes.titlesize':   12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.framealpha': 0.85,
    'figure.dpi':       150,
})

MODEL_COLORS = {
    'GPAW':      '#222222',
    'MACE':      '#2196F3',
    'CHGNet':    '#F44336',
    'TensorNet': '#4CAF50',
}
MODEL_MARKERS = {
    'GPAW': 'o', 'MACE': 's', 'CHGNet': '^', 'TensorNet': 'D'
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    return p.parse_args()


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_gpaw(sizes):
    """Load GPAW sites for given sizes. Returns {site_id: dict}."""
    sites = {}
    for sz in sizes:
        cid = f'Cu{sz}_00'
        r = GPAW_DIR / f'Cu{sz}' / f'results_{cid}.json'
        c = GPAW_DIR / f'Cu{sz}' / f'checkpoint_{cid}.json'
        f = r if r.exists() else c if c.exists() else None
        if not f:
            continue
        d = json.load(open(f))
        raw = d.get('sites', d.get('site_results', []))
        for s in raw:
            if s.get('dgh_gpaw_eV') is not None:
                sites[s['site_global_id']] = s
    return sites


def load_mlip(fname, key):
    path = RESULTS / fname
    if not path.exists():
        return {}
    d = json.load(open(path))
    return {s['site_global_id']: s[key]
            for s in d['sites'] if s.get(key) is not None}


def load_prepared():
    """Load prepared site geometries {site_id: dict with cluster_pos, h_pos}."""
    with open(GPAW_DIR / 'sites_prepared.json') as f:
        data = json.load(f)
    return {s['site_global_id']: s for s in data}


# ── Figure 5: CHE Free Energy Diagrams ───────────────────────────────────────

def draw_che_step(ax, dgh, color, label=None, lw=2.0, alpha=1.0, offset=0.0):
    """
    Draw a 2-step CHE free energy staircase for one site/model.
    Steps: (H⁺+e⁻) [G=0] → H* [G=ΔGH*] → ½H₂ [G=0]
    x coordinates: 0-1 = adsorption step, 1-2 = desorption step
    offset: small horizontal nudge to separate overlapping lines
    """
    xs = [0.0 + offset, 1.0 + offset, 1.0 + offset, 2.0 + offset]
    ys = [0.0, 0.0, float(dgh), float(dgh)]
    ax.plot(xs[:2], ys[:2], color=color, lw=lw, alpha=alpha)   # step 1
    ax.plot([1.0+offset, 1.0+offset], [0.0, float(dgh)],
            color=color, lw=lw*0.6, alpha=alpha, ls='--')       # vertical
    xs2 = [1.0 + offset, 2.0 + offset]
    ys2 = [float(dgh), float(dgh)]
    ax.plot(xs2, ys2, color=color, lw=lw, alpha=alpha,
            label=label if label else None)
    # Final drop to 0
    ax.plot([2.0+offset, 2.0+offset], [float(dgh), 0.0],
            color=color, lw=lw*0.6, alpha=alpha, ls='--')
    ax.plot([2.0+offset, 3.0+offset], [0.0, 0.0],
            color=color, lw=lw, alpha=alpha)


def fig_che_free_energy(gpaw, mace_d, chgnet_d, tensornet_d, sizes):
    """
    One panel per cluster size. Each panel shows the CHE free energy
    staircase for the best GPAW site, overlaid with MLIP predictions
    for the same site.
    Also shows light gray lines for all other sites (full landscape).
    """
    n = len(sizes)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sz in zip(axes, sizes):
        # All GPAW sites for this size
        sz_sites = {k: v for k, v in gpaw.items() if v['size'] == sz}
        if not sz_sites:
            ax.set_title(f'Cu{sz}\n(no data)')
            continue

        # Background: all sites as thin grey lines
        all_dgh = [v['dgh_gpaw_eV'] for v in sz_sites.values()]
        for dgh in all_dgh:
            xs = [0, 1, 1, 2, 2, 3]
            ys = [0, 0, dgh, dgh, 0, 0]
            ax.plot([0,1], [0,0], color='#cccccc', lw=0.8, alpha=0.5)
            ax.plot([1,2], [dgh,dgh], color='#cccccc', lw=0.8, alpha=0.5)
            ax.plot([2,3], [0,0], color='#cccccc', lw=0.8, alpha=0.5)
            ax.plot([1,1], [0,dgh], color='#cccccc', lw=0.5, ls='--', alpha=0.4)
            ax.plot([2,2], [dgh,0], color='#cccccc', lw=0.5, ls='--', alpha=0.4)

        # Best site (min |ΔGH*|)
        best_id = min(sz_sites, key=lambda k: abs(sz_sites[k]['dgh_gpaw_eV']))
        best = sz_sites[best_id]
        best_dgh_gpaw = best['dgh_gpaw_eV']

        # Draw GPAW best site
        stype = best['site_type']
        draw_che_step(ax, best_dgh_gpaw, MODEL_COLORS['GPAW'],
                      label=f'GPAW ({stype})', lw=2.5)

        # Draw MLIPs for same site
        mlip_info = [
            ('MACE',      mace_d.get(best_id)),
            ('CHGNet',    chgnet_d.get(best_id)),
            ('TensorNet', tensornet_d.get(best_id)),
        ]
        offsets = [-0.04, 0.0, 0.04]
        for (name, dgh), off in zip(mlip_info, offsets):
            if dgh is not None:
                draw_che_step(ax, dgh, MODEL_COLORS[name],
                              label=f'{name}', lw=1.8, alpha=0.85, offset=off)

        # Thermoneutral reference line
        ax.axhline(0, color='black', lw=0.8, ls=':', alpha=0.4)

        # Annotations
        ax.text(1.5, best_dgh_gpaw + 0.02, f'{best_dgh_gpaw:+.3f} eV',
                ha='center', va='bottom', fontsize=8.5,
                color=MODEL_COLORS['GPAW'], fontweight='bold')

        ax.set_xlim(-0.2, 3.5)
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(['H⁺+e⁻\n→ H*', 'H*', 'H*\n→ ½H₂'], fontsize=9)
        ax.set_title(f'Cu{sz}  ({len(sz_sites)} sites)', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25)

        if ax == axes[0]:
            ax.set_ylabel('Free Energy ΔG (eV)')
            ax.legend(fontsize=7.5, loc='upper right',
                      handlelength=1.5, labelspacing=0.3)

    fig.suptitle('CHE Free Energy Diagram — Best HER Site per Cluster Size\n'
                 '(grey = all sites, colored = best GPAW site + MLIP predictions)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = VIZ_DIR / 'fig5_che_free_energy.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: Structure Gallery ───────────────────────────────────────────────

def render_cluster(ax, cluster_pos, h_pos, site_atoms, site_type,
                   title='', dgh_gpaw=None, dgh_mace=None):
    """
    Render a Cu nanocluster + H adsorption site in 3D matplotlib.
    Atoms sorted by z so front atoms draw on top (painter's algorithm).
    Cu atoms colored copper-tone by depth; site atoms gold; H pink.
    """
    cluster_pos = np.array(cluster_pos)
    h_pos       = np.array(h_pos)
    center      = cluster_pos.mean(axis=0)
    n           = len(cluster_pos)

    # View direction: rotate so H site faces viewer
    h_dir  = h_pos - center
    elev   = float(np.degrees(np.arcsin(
                 np.clip(h_dir[2] / (np.linalg.norm(h_dir) + 1e-9), -1, 1))))
    azim   = float(np.degrees(np.arctan2(h_dir[1], h_dir[0])))
    ax.view_init(elev=max(15, min(45, elev)), azim=azim - 20)

    # Atom radii (Å → plot units)
    cu_r = max(120, 1200 // n)
    h_r  = cu_r * 0.5

    # Depth-sort: draw back atoms first so front ones overlap them
    # Project onto view direction to get depth
    view_vec = h_dir / (np.linalg.norm(h_dir) + 1e-9)
    depths   = cluster_pos @ view_vec
    order    = np.argsort(depths)   # back → front

    site_set = set(site_atoms) if site_atoms else set()

    for idx in order:
        pos = cluster_pos[idx]
        depth_norm = (depths[idx] - depths.min()) / (depths.max() - depths.min() + 1e-9)

        if idx in site_set:
            color     = '#FFD700'   # gold
            edge      = '#B8860B'
            size      = cu_r * 1.5
        else:
            # Copper tone: dark core → bright surface
            r = int(180 + 40 * depth_norm)
            g = int(90  + 40 * depth_norm)
            b = int(20)
            color = f'#{r:02x}{g:02x}{b:02x}'
            edge  = '#5c2d00'
            size  = cu_r

        ax.scatter(*pos, c=color, s=size, depthshade=True,
                   edgecolors=edge, linewidths=0.6, zorder=2)

    # Draw Cu–Cu bonds for nearest neighbors (distance < 3.2 Å)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(cluster_pos[i] - cluster_pos[j])
            if d < 3.2:
                xs = [cluster_pos[i,0], cluster_pos[j,0]]
                ys = [cluster_pos[i,1], cluster_pos[j,1]]
                zs = [cluster_pos[i,2], cluster_pos[j,2]]
                ax.plot(xs, ys, zs, color='#8B4513', lw=0.5, alpha=0.35, zorder=1)

    # Bond from H to nearest Cu
    dists_to_cu = np.linalg.norm(cluster_pos - h_pos, axis=1)
    nearest     = np.argmin(dists_to_cu)
    bond_end    = cluster_pos[nearest]
    ax.plot([h_pos[0], bond_end[0]],
            [h_pos[1], bond_end[1]],
            [h_pos[2], bond_end[2]],
            color='#555555', lw=1.2, alpha=0.7, zorder=3)

    # H atom (draw last = on top)
    ax.scatter(*h_pos, c='#FF69B4', s=h_r * 1.4,
               edgecolors='#CC1166', linewidths=1.2, zorder=5,
               depthshade=False)

    ax.set_axis_off()

    # Tight view box
    all_pts  = np.vstack([cluster_pos, h_pos])
    pad      = 2.5
    ax.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
    ax.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
    ax.set_zlim(all_pts[:,2].min()-pad, all_pts[:,2].max()+pad)

    # Title
    title_str = title
    if dgh_gpaw is not None:
        title_str += f'\nGPAW: {dgh_gpaw:+.3f} eV'
    if dgh_mace is not None:
        title_str += f'  MACE: {dgh_mace:+.3f} eV'
    ax.set_title(title_str, fontsize=9, pad=4)


def fig_structure_gallery(gpaw, prepared, sizes):
    """
    One 3D panel per cluster size showing the best GPAW site.
    """
    n = len(sizes)
    fig = plt.figure(figsize=(3.5*n, 4.5))

    for i, sz in enumerate(sizes):
        ax = fig.add_subplot(1, n, i+1, projection='3d')

        sz_sites = {k: v for k, v in gpaw.items() if v['size'] == sz}
        if not sz_sites:
            ax.set_title(f'Cu{sz}\n(no data)')
            continue

        # Best site
        best_id  = min(sz_sites, key=lambda k: abs(sz_sites[k]['dgh_gpaw_eV']))
        best     = sz_sites[best_id]
        prep     = prepared.get(best_id)

        if prep is None:
            ax.set_title(f'Cu{sz}\n(geometry missing)')
            continue

        render_cluster(
            ax,
            cluster_pos = prep['cluster_pos'],
            h_pos       = prep['h_pos'],
            site_atoms  = best['site_atoms'],
            site_type   = best['site_type'],
            title       = f'Cu{sz}  ({best["site_type"]})',
            dgh_gpaw    = best['dgh_gpaw_eV'],
            dgh_mace    = best['dgh_mace_eV'],
        )

    # Legend elements
    cu_patch = mpatches.Patch(facecolor='#f59e0b', edgecolor='#7c3d00',
                               label='Cu (surface)')
    cu_core  = mpatches.Patch(facecolor='#b45309', edgecolor='#7c3d00',
                               label='Cu (core)')
    site_patch = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                                 label='Site atoms')
    h_patch  = mpatches.Patch(facecolor='#ff69b4', edgecolor='#cc1166',
                               label='H* adsorbed')

    fig.legend(handles=[cu_patch, cu_core, site_patch, h_patch],
               loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('MACE-Relaxed Structures — Best HER Site (min |ΔGH*|) per Cluster Size\n'
                 'Gold = site atoms, Pink = H*, Color gradient = surface → core',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = VIZ_DIR / 'fig6_structure_gallery.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 7: ΔGH* Landscape (pseudo-volcano) ────────────────────────────────

def fig_dgh_landscape(gpaw, mace_d, chgnet_d, tensornet_d, sizes):
    """
    For each cluster size, show ΔGH* values as horizontal tick marks
    for all sites and all models. Vertical layout = size. This shows
    the full adsorption energy landscape and where each model places
    sites relative to thermoneutral (ΔGH*=0).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    model_data = [
        ('GPAW',      {k: v['dgh_gpaw_eV'] for k, v in gpaw.items()},
         MODEL_COLORS['GPAW'], 6.0, 0.12),
        ('MACE',      mace_d, MODEL_COLORS['MACE'],      4.0, 0.08),
        ('CHGNet',    chgnet_d, MODEL_COLORS['CHGNet'],  3.0, 0.06),
        ('TensorNet', tensornet_d, MODEL_COLORS['TensorNet'], 3.0, 0.06),
    ]

    y_ticks, y_labels = [], []
    y_offsets = {'GPAW': 0.15, 'MACE': 0.05, 'CHGNet': -0.05, 'TensorNet': -0.15}

    for i, sz in enumerate(sizes):
        y_base = i * 1.0
        y_ticks.append(y_base)
        y_labels.append(f'Cu{sz}')

        for name, vals_dict, color, lw, height in model_data:
            sz_vals = [v for sid, v in vals_dict.items()
                       if sz in (gpaw[sid]['size'] if sid in gpaw else 0,)]

            # For non-GPAW models, match to GPAW site IDs for this size
            if name != 'GPAW':
                sz_vals = [v for sid, v in vals_dict.items()
                           if sid in gpaw and gpaw[sid]['size'] == sz]

            y = y_base + y_offsets[name]
            for v in sz_vals:
                ax.plot([v, v], [y - height/2, y + height/2],
                        color=color, lw=lw, alpha=0.7, solid_capstyle='round')

    # Thermoneutral line
    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.5,
               label='Thermoneutral (ΔGH*=0)')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel('ΔGH* (eV)', fontsize=12)
    ax.set_title('HER ΔGH* Landscape — All Sites per Cluster Size\n'
                 '(each tick = one adsorption site; closer to 0 = better HER activity)',
                 fontweight='bold')
    ax.set_xlim(-0.7, 0.7)
    ax.grid(True, axis='x', alpha=0.25)

    # Legend
    handles = [mpatches.Patch(color=MODEL_COLORS[n], label=n)
               for n in ['GPAW', 'MACE', 'CHGNet', 'TensorNet']]
    handles.append(plt.Line2D([0], [0], color='black', ls='--', lw=1.2,
                               label='Thermoneutral'))
    ax.legend(handles=handles, fontsize=9, loc='upper right')

    fig.tight_layout()
    out = VIZ_DIR / 'fig7_dgh_landscape.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sizes = sorted(args.sizes)

    print("=" * 65)
    print("08 — Publication Figures (CHE diagrams, structures, landscape)")
    print(f"  Sizes: {sizes}")
    print("=" * 65)

    print("\nLoading data...")
    gpaw      = load_gpaw(sizes)
    mace_d    = {k: v['dgh_mace_eV'] for k, v in gpaw.items()}
    chgnet_d  = load_mlip('chgnet/results_chgnet.json',    'dgh_chgnet_eV')
    tensornet_d = load_mlip('tensornet/results_tensornet.json', 'dgh_tensornet_eV')
    prepared  = load_prepared()

    print(f"  GPAW:      {len(gpaw)} sites")
    print(f"  CHGNet:    {len(chgnet_d)} sites")
    print(f"  TensorNet: {len(tensornet_d)} sites")
    print(f"  Prepared:  {len(prepared)} geometries")

    print("\nGenerating figures...")

    print("  Fig 5: CHE free energy diagrams...")
    fig_che_free_energy(gpaw, mace_d, chgnet_d, tensornet_d, sizes)

    print("  Fig 6: Structure gallery...")
    fig_structure_gallery(gpaw, prepared, sizes)

    print("  Fig 7: ΔGH* landscape...")
    fig_dgh_landscape(gpaw, mace_d, chgnet_d, tensornet_d, sizes)

    print(f"\nAll figures saved to viz/:")
    for f in sorted(VIZ_DIR.glob('fig*.png')):
        print(f"  {f.name}")
    print("\nRe-run 07_compare_results.py to update fig1–4 with latest GPAW data.")


if __name__ == '__main__':
    main()
