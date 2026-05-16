# Dataset Description

## dft_reference/

Per-cluster GPAW PBE-D3(BJ) reference calculations. Each folder contains results for
one cluster size across three independent geometries (_00, _01, _02).

### File: `results_CuN_XX.json`

List of per-site dictionaries. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `site_id` | str | Unique site identifier (e.g. `"site_000_bridge"`) |
| `site_type` | str | `"atop"`, `"bridge"`, or `"hollow"` |
| `cluster_id` | str | Cluster identifier (e.g. `"Cu40_01"`) |
| `dgh_gpaw_eV` | float | DFT ΔG_H* (eV), PBE-D3(BJ) reference |
| `dgh_mace_eV` | float | MACE-MP-0 zero-shot ΔG_H* (eV) |
| `dgh_chgnet_eV` | float | CHGNet v0.3.0 zero-shot ΔG_H* (eV) |
| `dgh_tensornet_eV` | float | TensorNet MatPES-PBE-v2025.1 zero-shot ΔG_H* (eV) |
| `forces_gpaw` | list | DFT forces on H atom [eV/Å] |
| `h_position` | list | Cartesian coordinates of H* [Å] |

Fine-tuning prediction columns (where available):
`dgh_mace_ft_N{N}_seed{s}_eV` — MACE fine-tuned on N training configs, seed s.

### File: `cluster_CuN_XX.txt`

ASE-compatible XYZ/extxyz file of the MACE-MP-0 relaxed bare cluster geometry.

---

## analysis/

Pre-computed summary statistics from the benchmark scripts.

| File | Contents |
|------|----------|
| `benchmark_summary.json` | Zero-shot MAE, RMSE, Spearman ρ, top-k% recall per model |
| `learning_curves_ext.json` | MACE FT MAE vs N (10–100), mean ± std over 3 seeds, 364-site dataset |
| `rank_order_lc_ext.json` | Spearman ρ and top-10/15/20/25% recall vs N, both MACE and CHGNet |
| `isomerism_results.json` | ΔG_H* spread across _00/_01/_02 geometries per cluster size |
| `bootstrap_ci_results.json` | 95% bootstrap CIs on Spearman ρ values in Table 4 |
| `chgnet_ro_lc.json` | CHGNet fine-tuning rank-order learning curve |

---

## Units and Sign Convention

- All ΔG_H* values in **eV**
- **Negative ΔG_H*** = H* binds more strongly than ½H₂ (over-binding)
- **Optimal HER activity:** ΔG_H* ≈ 0 eV (thermoneutral)
- ZPE + entropy correction: +0.24 eV added to all raw adsorption energies

## Reference

Tavana, J. *Benchmarking Foundation ML Interatomic Potentials for HER Activity
Screening of Copper Nanoclusters.* ACS Catalysis (2026).
