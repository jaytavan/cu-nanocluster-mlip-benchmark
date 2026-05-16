"""
04b_gpaw_launch.py

Parallel GPAW Launcher
-----------------------
Runs 04b_gpaw_worker.py for all 5 cluster sizes simultaneously (or sequentially
with --sequential to avoid RAM exhaustion on 32 GB systems).

Core allocation:
  5 workers × NCORES_EACH OMP threads
  Default: 5 × 4 = 20 cores  (leaves 4 of 24 free for OS)
  Use --ncores 4 (default) for balanced load.

Why not 5 × 5 = 25? The i9-13900HX has 24 cores — stay within budget
to avoid scheduling contention. 4 threads/job is empirically good for
GPAW PW mode (matching prior project configuration).

Memory note: GPAW peaks at ~6–8 GB/worker during wavefunction init for Cu50.
Running all 5 sizes in parallel requires ~20–25 GB peak. Use --sequential
(or --ncores 12 with one size at a time) when RAM is under pressure.

Usage:
    python3 04b_gpaw_launch.py                           # full run (cluster 00)
    python3 04b_gpaw_launch.py --test                    # 2 sites/size (sanity check)
    python3 04b_gpaw_launch.py --sizes 50                # run only Cu50
    python3 04b_gpaw_launch.py --ncores 4                # 4 OMP per worker
    python3 04b_gpaw_launch.py --cluster-ids 01 02       # Phase C: extra geometries
    python3 04b_gpaw_launch.py --sequential              # one size at a time (low RAM)

Output:
    results/gpaw/Cu{N}/results_Cu{N}_{ID}.json   per size/cluster-id
    results/gpaw/summary_mace_vs_gpaw.json        aggregated MAE/RMSE table
    results/gpaw/worker_Cu{N}_{ID}.log            stdout from each worker
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SIZES        = [10, 20, 30, 40, 50]
NCORES_EACH  = 4
PYTHON       = sys.executable
WORKER       = Path(__file__).parent / '04b_gpaw_worker.py'
RESULTS      = Path(__file__).parent.parent / 'results'
GPAW_DIR     = RESULTS / 'gpaw'
GPAW_DIR.mkdir(parents=True, exist_ok=True)

# Stagger launch by this many seconds to avoid filesystem races at startup
LAUNCH_DELAY = 5   # seconds between worker launches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true',
                   help='Run only 2 sites per size')
    p.add_argument('--sizes', type=int, nargs='+', default=SIZES,
                   help='Which sizes to run (default: all 5)')
    p.add_argument('--ncores', type=int, default=NCORES_EACH,
                   help='OMP threads per worker (default: 4)')
    p.add_argument('--cluster-ids', nargs='+', default=['00'],
                   help='Cluster geometry IDs to process (default: 00)')
    p.add_argument('--sequential', action='store_true',
                   help='Run one size at a time (prevents OOM on 32 GB systems)')
    return p.parse_args()


def launch_worker(size: int, cluster_id: str, ncores: int, test: bool, log_path: Path):
    """Start one worker subprocess, redirect stdout+stderr to log file."""
    cmd = [PYTHON, str(WORKER), '--size', str(size),
           '--cluster-id', cluster_id, '--ncores', str(ncores)]
    if test:
        cmd.append('--test')

    env = os.environ.copy()
    env['OMP_NUM_THREADS']      = str(ncores)
    env['OPENBLAS_NUM_THREADS'] = str(ncores)
    env['MKL_NUM_THREADS']      = str(ncores)

    log_fh = open(log_path, 'w')
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        env=env, text=True,
    )
    return proc, log_fh


def load_result(size: int, cluster_id: str = '00'):
    """Load result JSON for one size/cluster_id if it exists."""
    f = GPAW_DIR / f'Cu{size}' / f'results_Cu{size}_{cluster_id}.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def print_status(procs: dict, start_time: float):
    elapsed = (time.time() - start_time) / 60
    print(f"\n[{elapsed:.1f} min elapsed]  Worker status:")

    for (sz, cid), (proc, _) in sorted(procs.items()):
        rc = proc.poll()
        if rc is None:
            state = 'running'
        elif rc == 0:
            state = 'done ✓'
        else:
            state = f'FAILED (exit {rc})'

        extra = ''
        if rc == 0:
            r = load_result(sz, cid)
            if r and r.get('mae_eV') is not None:
                extra = (f"  MAE={r['mae_eV']:.4f} eV  "
                         f"RMSE={r['rmse_eV']:.4f} eV  "
                         f"({r['n_computed']}/{r['n_sites_total']} sites)")
        elif rc is None:
            ckpt = GPAW_DIR / f'Cu{sz}' / f'checkpoint_Cu{sz}_{cid}.json'
            if ckpt.exists():
                try:
                    with open(ckpt) as f:
                        c = json.load(f)
                    done = sum(1 for r in c['site_results'] if r.get('dgh_gpaw_eV') is not None)
                    extra = f'  {done} sites done'
                except Exception:
                    pass

        print(f"  Cu{sz}_{cid}  {state:<15}{extra}")

    return {key: procs[key][0].poll() for key in procs}


def aggregate_results(sizes: list, cluster_ids: list) -> dict:
    """Collect all worker results → summary JSON and print table."""
    summary = {}
    for sz in sizes:
        for cid in cluster_ids:
            r = load_result(sz, cid)
            key = f'Cu{sz}_{cid}'
            if r:
                summary[key] = {
                    'config_id':     r['config_id'],
                    'n_computed':    r['n_computed'],
                    'n_failed':      r['n_failed'],
                    'dgh_gpaw_min':  r['dgh_gpaw_min'],
                    'dgh_gpaw_max':  r['dgh_gpaw_max'],
                    'dgh_mace_min':  r['dgh_mace_min'],
                    'dgh_mace_max':  r['dgh_mace_max'],
                    'mae_eV':        r['mae_eV'],
                    'rmse_eV':       r['rmse_eV'],
                    'total_time_s':  r['total_time_s'],
                }

    out = GPAW_DIR / 'summary_mace_vs_gpaw.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*75}")
    print("MACE vs GPAW — Summary")
    print(f"{'='*75}")
    print(f"  {'Config':<12}  {'Sites':>6}  "
          f"{'ΔGH GPAW':>12}  {'ΔGH MACE':>12}  "
          f"{'MAE (eV)':>10}  {'RMSE (eV)':>11}")
    print("  " + "-" * 69)

    for key in sorted(summary):
        s = summary[key]
        if s['mae_eV'] is None:
            continue
        gpaw_range = f"[{s['dgh_gpaw_min']:+.2f},{s['dgh_gpaw_max']:+.2f}]"
        mace_range = f"[{s['dgh_mace_min']:+.2f},{s['dgh_mace_max']:+.2f}]"
        print(f"  {key:<12}  {s['n_computed']:>6}  "
              f"{gpaw_range:>12}  {mace_range:>12}  "
              f"{s['mae_eV']:>10.4f}  {s['rmse_eV']:>11.4f}")

    if summary:
        all_mae  = [s['mae_eV']  for s in summary.values() if s['mae_eV']  is not None]
        all_rmse = [s['rmse_eV'] for s in summary.values() if s['rmse_eV'] is not None]
        import numpy as np
        print("  " + "-" * 69)
        print(f"  {'OVERALL':<12}  {'':>6}  {'':>12}  {'':>12}  "
              f"{np.mean(all_mae):>10.4f}  {np.mean(all_rmse):>11.4f}")

    print(f"\n  Full results: {out}")
    print(f"  Next: run 05_chgnet_dgh.py and 06_tensornet_dgh.py")
    return summary


def run_parallel(sizes, cluster_ids, args):
    """Launch all workers simultaneously and monitor until done."""
    n_workers = len(sizes) * len(cluster_ids)
    print(f"  Workers:     {n_workers}  ×  {args.ncores} OMP threads")
    print(f"  Total cores: {n_workers * args.ncores} / 24 available on i9\n")

    procs = {}
    for sz in sizes:
        for cid in cluster_ids:
            log_path = GPAW_DIR / f'worker_Cu{sz}_{cid}.log'
            proc, fh = launch_worker(sz, cid, args.ncores, args.test, log_path)
            procs[(sz, cid)] = (proc, fh)
            print(f"  Launched Cu{sz}_{cid}  PID={proc.pid}  log → {log_path.name}")
            time.sleep(LAUNCH_DELAY)

    print(f"\nAll {n_workers} workers launched. Monitoring every 5 min...")
    print(f"  Live logs:  tail -f {GPAW_DIR}/worker_Cu*.log")
    pids = ' '.join(str(p.pid) for p, _ in procs.values())
    print(f"  Kill all:   kill {pids}\n")

    start_time = time.time()
    while True:
        time.sleep(300)
        status = print_status(procs, start_time)
        if all(v is not None for v in status.values()):
            break

    for _, (_, fh) in procs.items():
        if fh:
            fh.close()

    wall_min = (time.time() - start_time) / 60
    print(f"\nAll parallel workers finished.  Wall time: {wall_min:.1f} min")
    failed = [(sz, cid) for (sz, cid), (p, _) in procs.items() if p.poll() != 0]
    if failed:
        print(f"WARNING: Workers failed for: {failed}")
    return failed


def run_sequential(sizes, cluster_ids, args):
    """Run one worker at a time — safe on 32 GB RAM systems."""
    print(f"  Mode: SEQUENTIAL (one size at a time to limit peak RAM)\n")
    failed = []
    start_all = time.time()
    for sz in sizes:
        for cid in cluster_ids:
            log_path = GPAW_DIR / f'worker_Cu{sz}_{cid}.log'
            print(f"  Starting Cu{sz}_{cid}  log → {log_path.name}")
            t0 = time.time()
            proc, fh = launch_worker(sz, cid, args.ncores, args.test, log_path)
            procs_one = {(sz, cid): (proc, fh)}
            while proc.poll() is None:
                time.sleep(300)
                print_status(procs_one, t0)
            if fh:
                fh.close()
            elapsed = (time.time() - t0) / 60
            rc = proc.poll()
            outcome = 'done ✓' if rc == 0 else f'FAILED (exit {rc})'
            print(f"  Cu{sz}_{cid} {outcome}  [{elapsed:.1f} min]")
            if rc != 0:
                failed.append((sz, cid))

    wall_min = (time.time() - start_all) / 60
    print(f"\nAll sequential workers finished.  Total wall time: {wall_min:.1f} min")
    if failed:
        print(f"WARNING: Workers failed for: {failed}")
    return failed


def main():
    args = parse_args()
    sizes = sorted(args.sizes)
    cluster_ids = args.cluster_ids

    mode = 'TEST (2 sites each)' if args.test else 'FULL'
    exec_mode = 'SEQUENTIAL' if args.sequential else 'PARALLEL'
    print(f"{'='*65}")
    print(f"GPAW {'Sequential' if args.sequential else 'Parallel'} Launcher")
    print(f"  Mode:        {mode}")
    print(f"  Execution:   {exec_mode}")
    print(f"  Sizes:       {sizes}")
    print(f"  Cluster IDs: {cluster_ids}")
    print(f"  OMP threads: {args.ncores} per worker")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    # Check prep was done
    prep_json = GPAW_DIR / 'sites_prepared.json'
    h2_cache  = GPAW_DIR / 'h2_energy_gpaw.json'
    if not prep_json.exists() or not h2_cache.exists():
        print("ERROR: Run 04a_gpaw_prep.py first.")
        sys.exit(1)

    if args.sequential:
        failed = run_sequential(sizes, cluster_ids, args)
    else:
        failed = run_parallel(sizes, cluster_ids, args)

    print(f"\n{'='*65}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if failed:
        print(f"  Check logs: {GPAW_DIR}/worker_Cu*.log")

    aggregate_results(sizes, cluster_ids)


if __name__ == '__main__':
    main()
