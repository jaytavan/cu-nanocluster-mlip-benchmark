"""
08b_structure_gallery_ovito.py

Publication-quality structure gallery using OVITO + Tachyon ray-tracer.
Renders the best HER site (min |ΔGH*|) for each cluster size.

Output:
  viz/struct_Cu10.png … viz/struct_Cu50.png   ← per-size renders
  viz/fig6_structure_gallery.png               ← combined panel (replaces matplotlib version)

Color scheme:
  Cu (background):  copper brown  (#C87533)
  Cu (site atoms):  gold          (#FFD700)
  H* (adsorbed):    pink          (#FF69B4)
  Background:       white

Usage:
    python3 08b_structure_gallery_ovito.py
    python3 08b_structure_gallery_ovito.py --sizes 10 20 30 40
"""

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import argparse
import json
import numpy as np
from pathlib import Path
from ase import Atoms

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import StaticSource, Pipeline
from ovito.vis import Viewport, TachyonRenderer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).parent.parent
RESULTS  = PROJECT / "results"
GPAW_DIR = RESULTS / "gpaw"
VIZ_DIR  = PROJECT / "viz"
VIZ_DIR.mkdir(exist_ok=True)

SIZES = [10, 20, 30, 40, 50]

# Camera elevation tilt added on top of the H*-facing direction (degrees).
# Positive = camera moves upward so you see the cluster from a 3/4 angle
# rather than looking straight at H* from the side, which obscures depth.
H_CAMERA_TILT_DEG = 25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    return p.parse_args()



def render_structure(size, cluster_pos, h_pos, site_atoms, dgh_gpaw, dgh_mace,
                     site_type, out_path):
    """Render one cluster+H structure with OVITO Tachyon renderer."""
    n = len(cluster_pos)
    site_set = set(site_atoms)

    atoms = Atoms(['Cu']*n + ['H'],
                  positions=np.vstack([cluster_pos, h_pos]))
    atoms.set_pbc(False)
    atoms.center(vacuum=3.5)

    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))

    # Hide simulation cell
    pipeline.source.data.cell.vis.enabled = False

    # Per-particle colors and radii
    def set_appearance(frame, data):
        colors = data.particles_.create_property('Color')
        for i in range(n):
            if i in site_set:
                colors[i] = (1.00, 0.84, 0.00)   # gold — site atoms
            else:
                colors[i] = (0.78, 0.50, 0.20)   # copper brown
        colors[n] = (1.00, 0.41, 0.71)            # hot pink — H

        radii = data.particles_.create_property('Radius')
        radii[:n] = 1.28    # Cu van der Waals / covalent radius
        radii[n]  = 0.75    # H slightly enlarged for visibility

    pipeline.modifiers.append(set_appearance)
    pipeline.add_to_scene()

    # Camera: always face H* directly, then tilt upward for depth.
    # Step 1 — vector from cluster centre to H* (the "face H*" direction)
    cluster_center = np.mean(cluster_pos, axis=0)
    h_vec = np.array(h_pos) - cluster_center
    h_dist = np.linalg.norm(h_vec)
    if h_dist < 1e-3:                          # degenerate: H at centre
        h_vec = np.array([0.0, 1.0, 0.0])
    h_hat = h_vec / np.linalg.norm(h_vec)     # unit vector toward H*

    # Step 2 — rotate that vector upward by H_CAMERA_TILT_DEG
    # Find a perpendicular "up" axis in the plane containing h_hat and world-Z
    world_z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(h_hat, world_z)) > 0.99:    # H* nearly straight up/down
        world_z = np.array([1.0, 0.0, 0.0])
    right = np.cross(h_hat, world_z)
    right /= np.linalg.norm(right)
    up = np.cross(right, h_hat)
    up /= np.linalg.norm(up)

    tilt = np.radians(H_CAMERA_TILT_DEG)
    cam_dir = h_hat * np.cos(tilt) + up * np.sin(tilt)
    cam_dir /= np.linalg.norm(cam_dir)

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.camera_dir = tuple(-cam_dir)   # camera points FROM outside TOWARD cluster
    vp.zoom_all()

    vp.render_image(
        filename=str(out_path),
        size=(600, 600),
        background=(1.0, 1.0, 1.0),
        renderer=TachyonRenderer(ambient_occlusion=True, shadows=True)
    )

    # Clean up scene for next render
    pipeline.remove_from_scene()
    print(f"  Cu{size}: saved {out_path.name}  "
          f"(ΔGH*={dgh_gpaw:+.3f} eV GPAW, {dgh_mace:+.3f} eV MACE, {site_type})")


def build_panel(sizes, render_paths):
    """Combine individual renders into one publication panel with matplotlib."""
    n = len(render_paths)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4.2))
    if n == 1:
        axes = [axes]

    gpaw_data = {}
    for sz in sizes:
        cid = f'Cu{sz}_00'
        rf = GPAW_DIR / f'Cu{sz}' / f'results_{cid}.json'
        cf = GPAW_DIR / f'Cu{sz}' / f'checkpoint_{cid}.json'
        f  = rf if rf.exists() else cf if cf.exists() else None
        if not f:
            continue
        d = json.load(open(f))
        sites = d.get('sites', d.get('site_results', []))
        good  = [s for s in sites if s.get('dgh_gpaw_eV') is not None]
        if good:
            gpaw_data[sz] = min(good, key=lambda s: abs(s['dgh_gpaw_eV']))

    for ax, (sz, path) in zip(axes, render_paths.items()):
        if path.exists():
            img = mpimg.imread(str(path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'pending', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')

        ax.set_axis_off()
        g = gpaw_data.get(sz)
        if g:
            title = (f'Cu$_{{{sz}}}$  ({g["site_type"]})\n'
                     f'ΔG$_{{H*}}$ = {g["dgh_gpaw_eV"]:+.3f} eV (GPAW)\n'
                     f'ΔG$_{{H*}}$ = {g["dgh_mace_eV"]:+.3f} eV (MACE)')
        else:
            title = f'Cu$_{{{sz}}}$\n(GPAW pending)'
        ax.set_title(title, fontsize=9, pad=4, linespacing=1.5)

    # Colored legend patches
    import matplotlib.patches as mpatches
    cu_patch   = mpatches.Patch(facecolor='#C87533', edgecolor='#7c3d00',
                                label='Cu atom')
    site_patch = mpatches.Patch(facecolor='#FFD700', edgecolor='#B8860B',
                                label='Site atoms (Cu)')
    h_patch    = mpatches.Patch(facecolor='#FF69B4', edgecolor='#CC1166',
                                label='H* adsorbed')
    fig.legend(handles=[cu_patch, site_patch, h_patch],
               loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True,
               title='Rendered with OVITO/Tachyon ray-tracer',
               title_fontsize=8)

    fig.suptitle('MACE-Relaxed Structures — Most Thermoneutral HER Site per Cluster Size',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout(pad=0.5)

    out = VIZ_DIR / 'fig6_structure_gallery.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Panel saved: {out}")


def main():
    args = parse_args()
    sizes = sorted(args.sizes)

    print("=" * 65)
    print("08b — Structure Gallery (OVITO Tachyon ray-tracer)")
    print(f"  Sizes: {sizes}")
    print("=" * 65)

    # Load data
    with open(GPAW_DIR / 'sites_prepared.json') as f:
        prep_list = json.load(f)
    prep_map = {s['site_global_id']: s for s in prep_list}

    render_paths = {}
    print()

    for sz in sizes:
        cid = f'Cu{sz}_00'
        rf  = GPAW_DIR / f'Cu{sz}' / f'results_{cid}.json'
        cf  = GPAW_DIR / f'Cu{sz}' / f'checkpoint_{cid}.json'
        f   = rf if rf.exists() else cf if cf.exists() else None

        out_path = VIZ_DIR / f'struct_Cu{sz}.png'
        render_paths[sz] = out_path

        if not f:
            print(f"  Cu{sz}: no GPAW data — skipping render")
            continue

        d     = json.load(open(f))
        sites = d.get('sites', d.get('site_results', []))
        good  = [s for s in sites if s.get('dgh_gpaw_eV') is not None]
        if not good:
            print(f"  Cu{sz}: no converged GPAW sites — skipping")
            continue

        best = min(good, key=lambda s: abs(s['dgh_gpaw_eV']))
        prep = prep_map.get(best['site_global_id'])
        if not prep:
            print(f"  Cu{sz}: geometry not in sites_prepared.json — skipping")
            continue

        render_structure(
            size        = sz,
            cluster_pos = np.array(prep['cluster_pos']),
            h_pos       = np.array(prep['h_pos']),
            site_atoms  = best['site_atoms'],
            dgh_gpaw    = best['dgh_gpaw_eV'],
            dgh_mace    = best['dgh_mace_eV'],
            site_type   = best['site_type'],
            out_path    = out_path,
        )

    print("\nBuilding combined panel...")
    build_panel(sizes, render_paths)
    print("\nDone. Individual renders in viz/struct_Cu*.png")


if __name__ == '__main__':
    main()
