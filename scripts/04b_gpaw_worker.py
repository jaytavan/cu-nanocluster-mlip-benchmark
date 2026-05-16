"""
04b_gpaw_worker.py  —  v2

GPAW ΔGH* Worker — one cluster size
-------------------------------------
Called by 04b_gpaw_launch.py. Runs GPAW PBE-D3(BJ)/PW350 single-points for
all prepared sites belonging to one cluster size (Cu10–Cu50).

Workflow per site:
  1. Load cluster+H geometry from sites_prepared.json (MACE-relaxed H pos)
  2. Run GPAW PBE/PW350 single-point → E(cluster+H), forces
  3. Add D3(BJ) dispersion correction to energy + forces
  4. ΔGH* = E(cluster+H) - E(cluster_GPAW) - ½·E(H2_GPAW) + 0.24 eV
  5. Save wavefunction to .gpw for future analysis (Bader, DOS, restarts)
     Note: E(cluster) is computed once per worker at startup (not per-site)

GPAW settings (v2):
  Functional:  PBE + D3(BJ) dispersion (dftd3 1.3.0 Python bindings)
  Basis:       plane-wave, 350 eV cutoff
  k-points:    Gamma-only (non-periodic cluster in vacuum)
  PAW:         GPAW defaults (Cu, H)
  Cell:        auto-centered with 10 Å vacuum each side
  SCF conv:    1e-5 eV  (tightened from v1 1e-4 for accurate forces)
  Wavefunction: saved to .gpw after each calculation

What changed vs v1:
  - Forces now computed and stored (for MACE fine-tuning)
  - D3(BJ) dispersion added to energy + forces
  - SCF convergence tightened: 1e-4 → 1e-5 eV
  - Wavefunction files (.gpw) saved for post-processing (Bader, DOS)

Geometry note:
  All calculations use MACE-relaxed cluster geometries (script 02) with
  MACE-relaxed H positions (script 04a). This is the standard approach for
  MLIP benchmarks — all models evaluate on the same geometry. The GPAW
  energies and forces are genuine DFT values at these geometries, not
  at the DFT-relaxed minimum. Explicitly stated in Methods section.

Checkpoint/resume: saves after every site — safe to Ctrl-C and restart.
.gpw files are written even on resume (overwrite-safe, same wavefunction).

Usage (called by 04b_gpaw_launch.py):
    python3 04b_gpaw_worker.py --size 50 --ncores 12 [--cluster-id 01] [--test]

Output per size:
    results/gpaw/Cu50/results_Cu50_00.json      ← final results (E + F + D3)
    results/gpaw/Cu50/checkpoint_Cu50_00.json   ← crash-safe checkpoint
    results/gpaw/Cu50/wavefunctions/            ← .gpw files (one per calc)
    results/gpaw/Cu50/gpaw_cluster.txt          ← GPAW log for cluster energy
    results/gpaw/Cu50/gpaw_site_*.txt           ← GPAW log per site
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path

# Parse args and set OMP BEFORE importing GPAW
parser = argparse.ArgumentParser()
parser.add_argument('--size',       type=int, required=True,
                    help='Cluster size (10, 20, 30, 40, or 50)')
parser.add_argument('--cluster-id', type=str, default='00',
                    help='Cluster geometry ID (default: 00)')
parser.add_argument('--ncores',     type=int, default=12,
                    help='OMP threads for GPAW (default: 12)')
parser.add_argument('--test',       action='store_true',
                    help='Process only 2 sites (sanity check)')
args = parser.parse_args()

os.environ['OMP_NUM_THREADS']      = str(args.ncores)
os.environ['OPENBLAS_NUM_THREADS'] = str(args.ncores)
os.environ['MKL_NUM_THREADS']      = str(args.ncores)

from ase import Atoms
from ase.calculators.mixing import SumCalculator
from gpaw import GPAW, PW, Mixer
from dftd3.ase import DFTD3

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).parent.parent
RESULTS   = PROJECT / "results"
GPAW_DIR  = RESULTS / "gpaw"
PREP_JSON = GPAW_DIR / "sites_prepared.json"
H2_CACHE  = GPAW_DIR / "h2_energy_gpaw.json"

SZ     = args.size
CID    = f"Cu{SZ}_{args.cluster_id}"
OUTDIR = GPAW_DIR / f"Cu{SZ}"
WFDIR  = OUTDIR / "wavefunctions"
OUTDIR.mkdir(parents=True, exist_ok=True)
WFDIR.mkdir(exist_ok=True)

# ── GPAW parameters ──────────────────────────────────────────────────────────
ECUT     = 350        # eV — plane-wave cutoff
SCF_CONV = 1e-5       # eV — tight convergence needed for accurate forces
ZPE_CORR = 0.24       # eV — CHE standard ZPE+entropy correction for H*


def make_gpaw(label: str) -> GPAW:
    """
    GPAW PBE/PW350 for nanocluster single-points.

    Settings:
      - Fermi width 0.2 eV: broader smearing smooths near-degenerate
        states at Fermi level in small clusters (< ~20 atoms).
      - Mixer beta=0.05: conservative Pulay mixing eliminates ±0.2 eV
        energy oscillations from default beta=0.25 in small clusters.
      - SCF conv 1e-5 eV: required for force accuracy (~1 meV/Å noise).
      - symmetry='off': needed for low-symmetry nanoclusters.
      - Wavefunction NOT saved here — call calc.write() explicitly after.
    """
    return GPAW(
        mode=PW(ECUT),
        xc='PBE',
        kpts={'gamma': True},
        occupations={'name': 'fermi-dirac', 'width': 0.2},
        mixer=Mixer(beta=0.05, nmaxold=5, weight=50),
        txt=str(OUTDIR / f'{label}.txt'),
        convergence={'energy': SCF_CONV},
        symmetry='off',
    )


def make_d3() -> DFTD3:
    """D3(BJ) dispersion correction for PBE (Grimme 2010)."""
    return DFTD3(method='PBE', damping='d3bj')


def gpaw_energy_forces(atoms: Atoms, label: str) -> dict:
    """
    Run GPAW PBE + D3(BJ) single-point.
    Returns dict with energy (eV), forces (eV/Å array), and paths.

    Energy:  E_PBE + E_D3
    Forces:  F_PBE + F_D3  (analytically from same SCF + D3 gradient)
    Wavefunction: saved to WFDIR/{label}.gpw
    """
    atoms = atoms.copy()
    atoms.set_pbc(False)
    atoms.center(vacuum=10.0)   # ~20×20×20 Å box

    gpaw_calc = make_gpaw(label)
    d3_calc   = make_d3()
    atoms.calc = SumCalculator([gpaw_calc, d3_calc])

    energy = float(atoms.get_potential_energy())   # PBE + D3
    forces = atoms.get_forces().tolist()            # PBE + D3 (same call chain)

    # Save GPAW wavefunction (PBE part only — the GPAW sub-calculator)
    gpw_path = WFDIR / f"{label}.gpw"
    gpaw_calc.write(str(gpw_path), mode='all')

    return {
        'energy_eV':  energy,
        'forces_eVA': forces,
        'gpw_path':   str(gpw_path),
    }


def rebuild_cluster_h(prepared: dict) -> Atoms:
    """Reconstruct cluster+H Atoms object from prepared dict."""
    cluster_pos = np.array(prepared['cluster_pos'])
    h_pos       = np.array(prepared['h_pos'])
    n_cu        = len(cluster_pos)
    symbols     = ['Cu'] * n_cu + ['H']
    all_pos     = np.vstack([cluster_pos, h_pos])
    atoms       = Atoms(symbols, positions=all_pos)
    atoms.set_pbc(False)
    return atoms


def rebuild_cluster(prepared: dict) -> Atoms:
    """Reconstruct bare cluster (no H) from prepared dict."""
    cluster_pos = np.array(prepared['cluster_pos'])
    n_cu        = len(cluster_pos)
    atoms       = Atoms(['Cu'] * n_cu, positions=cluster_pos)
    atoms.set_pbc(False)
    return atoms


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print(f"GPAW Worker v2 — Cu{SZ} ({CID})")
    print(f"  Functional:  PBE + D3(BJ)")
    print(f"  SCF conv:    {SCF_CONV} eV  (forces: yes)")
    print(f"  Wavefuncs:   {WFDIR}")
    print(f"  OMP threads: {args.ncores}")
    print(f"  Test mode:   {args.test}")
    print(f"{'='*65}\n")

    # ── Load prepared sites ──────────────────────────────────────────────────
    if not PREP_JSON.exists():
        raise FileNotFoundError(f"{PREP_JSON} not found — run 04a_gpaw_prep.py first")
    with open(PREP_JSON) as f:
        all_prepared = json.load(f)

    my_sites = [p for p in all_prepared if p['config_id'] == CID]
    if not my_sites:
        raise ValueError(f"No prepared sites found for {CID}")

    print(f"Found {len(my_sites)} prepared sites for {CID}")

    if args.test:
        my_sites = my_sites[:2]
        print(f"  TEST MODE: processing only {len(my_sites)} sites\n")

    # ── Load H2 reference ────────────────────────────────────────────────────
    if not H2_CACHE.exists():
        raise FileNotFoundError(f"{H2_CACHE} not found — run 04a_gpaw_prep.py first")
    with open(H2_CACHE) as f:
        h2_data = json.load(f)
    e_h2 = h2_data['e_h2_eV']
    print(f"E(H2) GPAW = {e_h2:.5f} eV  (from cache)")
    print(f"Note: H2 reference will be recomputed with D3 below.\n")

    # ── Step 0: Recompute H2 with D3 if not already cached ──────────────────
    h2_d3_cache = GPAW_DIR / "h2_energy_gpaw_d3.json"
    if h2_d3_cache.exists():
        e_h2 = json.load(open(h2_d3_cache))['e_h2_eV']
        print(f"E(H2) PBE+D3 = {e_h2:.5f} eV  (from cache)")
    else:
        print("Step 0: Computing E(H2) with PBE+D3...")
        from ase.build import molecule
        h2 = molecule('H2')
        h2.center(vacuum=6.0)
        h2.set_pbc(False)
        res = gpaw_energy_forces(h2, 'h2_d3')
        e_h2 = res['energy_eV']
        json.dump({'e_h2_eV': e_h2, 'xc': 'PBE+D3(BJ)'}, open(h2_d3_cache, 'w'), indent=2)
        print(f"  E(H2) PBE+D3 = {e_h2:.5f} eV")

    # ── Step 1: GPAW+D3 cluster energy ───────────────────────────────────────
    cluster_energy_file = OUTDIR / f'cluster_energy_{CID}_d3.json'
    if cluster_energy_file.exists():
        ce_data  = json.load(open(cluster_energy_file))
        e_cluster = ce_data['e_cluster_eV']
        f_cluster = ce_data['forces_eVA']
        print(f"E(cluster) PBE+D3 = {e_cluster:.5f} eV  (from cache)")
    else:
        print(f"Step 1: Computing GPAW+D3 E(cluster) for {CID}...")
        cluster = rebuild_cluster(my_sites[0])
        t0 = time.time()
        res = gpaw_energy_forces(cluster, f'cluster_{CID}')
        t_cluster = time.time() - t0
        e_cluster = res['energy_eV']
        f_cluster = res['forces_eVA']
        print(f"  E(cluster) = {e_cluster:.5f} eV  [{t_cluster:.1f}s]")
        json.dump({
            'config_id':    CID,
            'size':         SZ,
            'e_cluster_eV': e_cluster,
            'forces_eVA':   f_cluster,
            'gpw_path':     res['gpw_path'],
            'xc':           'PBE+D3(BJ)',
            'time_s':       t_cluster,
        }, open(cluster_energy_file, 'w'), indent=2)

    # ── Step 2: Site loop (with checkpoint/resume) ───────────────────────────
    checkpoint_file = OUTDIR / f'checkpoint_{CID}.json'
    if checkpoint_file.exists():
        ckpt         = json.load(open(checkpoint_file))
        site_results = ckpt['site_results']
        done_ids     = {r['site_global_id'] for r in site_results}
        print(f"\nResuming from checkpoint — {len(done_ids)} sites already done")
    else:
        site_results = []
        done_ids     = set()

    n_total     = len(my_sites)
    n_remaining = sum(1 for s in my_sites if s['site_global_id'] not in done_ids)
    print(f"\nStep 2: GPAW+D3 ΔGH* for {n_remaining} sites (of {n_total} total)")
    print(f"  {'#':>4}  {'ID':<22}  {'Type':<8}  {'ΔGH*':>8}  {'t(s)':>7}")
    print("  " + "-" * 58)

    t_start = time.time()

    for site in my_sites:
        sid = site['site_global_id']
        if sid in done_ids:
            continue

        t_site    = time.time()
        cluster_h = rebuild_cluster_h(site)
        site_idx  = site['site_idx']
        label     = f"site_{site_idx:03d}_{site['site_type']}"

        try:
            res      = gpaw_energy_forces(cluster_h, label)
            e_clus_h = res['energy_eV']
            f_clus_h = res['forces_eVA']
            delta_e  = e_clus_h - e_cluster - 0.5 * e_h2
            dgh_gpaw = delta_e + ZPE_CORR
            status   = 'ok'
        except Exception as ex:
            print(f"  FAILED: {sid}  error: {ex}")
            e_clus_h, f_clus_h, dgh_gpaw, status = None, None, None, f'error: {str(ex)[:80]}'
            res = {'gpw_path': None}

        elapsed = time.time() - t_site

        result = {
            'site_global_id': sid,
            'config_id':      CID,
            'size':           SZ,
            'site_idx':       site_idx,
            'site_type':      site['site_type'],
            'site_atoms':     site['site_atoms'],
            'dgh_mace_eV':    site['dgh_mace_eV'],
            'e_cluster_eV':   e_cluster,
            'e_h2_eV':        e_h2,
            'e_clus_h_eV':    round(float(e_clus_h), 6) if e_clus_h else None,
            'forces_eVA':     f_clus_h,            # list of [fx,fy,fz] per atom
            'dgh_gpaw_eV':    round(float(dgh_gpaw), 6) if dgh_gpaw else None,
            'delta_dgh_eV':   round(float(dgh_gpaw - site['dgh_mace_eV']), 6)
                              if dgh_gpaw else None,
            'gpw_path':       res.get('gpw_path'),
            'xc':             'PBE+D3(BJ)',
            'status':         status,
            'time_s':         round(elapsed, 1),
        }
        site_results.append(result)
        done_ids.add(sid)

        # Checkpoint after every site (crash-safe)
        ckpt = {
            'config_id':    CID,
            'size':         SZ,
            'xc':           'PBE+D3(BJ)',
            'e_cluster_eV': e_cluster,
            'e_h2_eV':      e_h2,
            'site_results': site_results,
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(ckpt, f)

        if dgh_gpaw is not None:
            n_done = len(done_ids)
            print(f"  {n_done:>4}  {sid:<22}  {site['site_type']:<8}  "
                  f"{dgh_gpaw:>8.4f}  {elapsed:>7.1f}")
        else:
            print(f"  ????  {sid:<22}  FAILED  {elapsed:>7.1f}")

    # ── Save final results ────────────────────────────────────────────────────
    good   = [r for r in site_results if r['dgh_gpaw_eV'] is not None]
    failed = [r for r in site_results if r['dgh_gpaw_eV'] is None]

    dgh_gpaw_vals = [r['dgh_gpaw_eV'] for r in good]
    dgh_mace_vals = [r['dgh_mace_eV'] for r in good]
    delta_vals    = [r['delta_dgh_eV'] for r in good]

    summary = {
        'config_id':      CID,
        'size':           SZ,
        'xc':             'PBE+D3(BJ)',
        'scf_conv_eV':    SCF_CONV,
        'forces_stored':  True,
        'wavefunctions':  str(WFDIR),
        'e_cluster_eV':   e_cluster,
        'e_h2_eV':        e_h2,
        'n_sites_total':  n_total,
        'n_computed':     len(good),
        'n_failed':       len(failed),
        'dgh_gpaw_min':   round(min(dgh_gpaw_vals), 6)             if good else None,
        'dgh_gpaw_max':   round(max(dgh_gpaw_vals), 6)             if good else None,
        'dgh_gpaw_mean':  round(float(np.mean(dgh_gpaw_vals)), 6)  if good else None,
        'dgh_mace_min':   round(min(dgh_mace_vals), 6)             if good else None,
        'dgh_mace_max':   round(max(dgh_mace_vals), 6)             if good else None,
        'mae_eV':         round(float(np.mean(np.abs(delta_vals))), 6) if good else None,
        'rmse_eV':        round(float(np.sqrt(np.mean(np.array(delta_vals)**2))), 6)
                          if good else None,
        'total_time_s':   round(time.time() - t_start, 1),
        'sites':          site_results,
    }

    out_file = OUTDIR / f'results_{CID}.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    wall = summary['total_time_s']
    print(f"\n{'='*65}")
    print(f"Cu{SZ} — DONE  (PBE+D3(BJ), forces stored, .gpw saved)")
    print(f"  Sites computed:  {len(good)} / {n_total}")
    print(f"  Failed:          {len(failed)}")
    if good:
        print(f"  ΔGH* GPAW:  [{min(dgh_gpaw_vals):.3f}, {max(dgh_gpaw_vals):.3f}] eV")
        print(f"  ΔGH* MACE:  [{min(dgh_mace_vals):.3f}, {max(dgh_mace_vals):.3f}] eV")
        print(f"  MAE  (MACE vs GPAW): {summary['mae_eV']:.4f} eV")
        print(f"  RMSE (MACE vs GPAW): {summary['rmse_eV']:.4f} eV")
    print(f"  Total time: {wall:.0f}s = {wall/3600:.2f} hr")
    print(f"  Results:    {out_file}")
    print(f"  Wavefuncs:  {WFDIR}/")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
