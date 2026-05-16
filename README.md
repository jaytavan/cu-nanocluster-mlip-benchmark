# Cu Nanocluster MLIP Benchmark

**First systematic benchmark of universal machine-learning interatomic potentials (MLIPs) for hydrogen adsorption free energy prediction on finite copper nanoclusters.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://doi.org/ZENODO_DOI_PLACEHOLDER)

---

## Overview

This repository accompanies the paper:

> Tavana, J. *Benchmarking Foundation ML Interatomic Potentials for HER Activity Screening of Copper Nanoclusters.* **ACS Catalysis** (submitted 2026). Preprint: arXiv:ARXIV_ID_PLACEHOLDER

We benchmark three universal MLIP foundation models — **MACE-MP-0**, **CHGNet**, and **TensorNet** — on predicting hydrogen adsorption free energy (ΔG_H*) for Cu nanoclusters (Cu₁₀–Cu₅₀), providing the first systematic evaluation on finite, non-periodic cluster systems for catalysis screening.

### Key findings

- Zero-shot MLIPs are insufficient for nanocluster HER screening: top-10% recall is **0% for MACE-MP-0 and CHGNet**, 11% for TensorNet
- Fine-tuning on **as few as 10 DFT calculations** recovers practical screening utility (recall 0.21 at N=10, 0.42 at N=40)
- GPU fine-tuning cost is **< 2% of DFT cost** — the bottleneck is always DFT, not the MLIP
- TensorNet is the best zero-shot choice (MAE 0.087 eV vs 0.181 eV for MACE, 0.406 eV for CHGNet)
- Cluster **isomerism contributes ΔG_H* spread of up to 0.21 eV** — geometry selection is a systematic uncertainty future benchmarks must address

---

## Repository Structure

```
cu-nanocluster-mlip-benchmark/
├── data/
│   ├── dft_reference/         ← 364 GPAW PBE-D3(BJ) ΔG_H* reference values
│   │   ├── Cu10/              ← results_Cu10_00/01/02.json + cluster geometries
│   │   ├── Cu20/
│   │   ├── Cu30/
│   │   ├── Cu40/
│   │   └── Cu50/
│   └── analysis/              ← pre-computed benchmark results
│       ├── benchmark_summary.json       ← zero-shot MAE, recall per model
│       ├── learning_curves_ext.json     ← MACE FT MAE vs N (364-site dataset)
│       ├── rank_order_lc_ext.json       ← Spearman ρ and recall vs N
│       ├── isomerism_results.json       ← ΔG_H* spread across cluster geometries
│       └── bootstrap_ci_results.json    ← bootstrap confidence intervals on Spearman ρ
├── scripts/
│   ├── 01_generate_clusters.py          ← FCC-based Cu nanocluster generation
│   ├── 02_mace_relax.py                 ← MACE-MP-0 geometry relaxation
│   ├── 03_adsorption_sites.py           ← bridge/hollow site enumeration
│   ├── 03b_atop_sites.py                ← atop site enumeration
│   ├── 04a_gpaw_prep.py                 ← GPAW input preparation
│   ├── 04b_gpaw_launch.py               ← parallel DFT launcher (checkpointed)
│   ├── 04b_gpaw_worker.py               ← per-site DFT worker
│   ├── 05_chgnet_dgh.py                 ← CHGNet zero-shot evaluation
│   ├── 06_tensornet_dgh.py              ← TensorNet zero-shot evaluation
│   ├── 07_compare_results.py            ← DFT vs MLIP comparison + figures
│   ├── 08_publication_figures.py        ← main publication figures
│   ├── 09_mace_finetune_learning_curves.py  ← MACE fine-tuning LC
│   ├── 09b_mace_ft_extended.py          ← extended LC to N=100
│   ├── 09c_reconstruct_ext_lc.py        ← reconstruct metrics from prediction arrays
│   ├── 10_rank_order_analysis.py        ← rank-order fidelity figures
│   ├── 11_bootstrap_ci.py               ← bootstrap CIs on Spearman ρ
│   ├── 12_rank_order_learning_curve.py  ← rank-order LC (original dataset)
│   ├── 13_chgnet_finetune_lc.py         ← CHGNet fine-tuning LC
│   ├── 14_cost_accuracy.py              ← DFT cost vs recall figure
│   ├── 17_isomerism_analysis.py         ← geometry sensitivity analysis
│   ├── 18_chgnet_ro_lc.py               ← CHGNet rank-order LC
│   ├── 19_regenerate_lc_figures.py      ← regenerate Figs 9 and 11
│   └── 19_workflow_schematic.py         ← Figure 1 workflow diagram
└── figures/                             ← all publication figures (PNG, 300 dpi)
```

---

## Dataset

The DFT reference dataset comprises **364 hydrogen adsorption configurations** computed with GPAW 25.7.0 at the PBE-D3(BJ) level:

- **Systems:** Cu₁₀, Cu₂₀, Cu₃₀, Cu₄₀, Cu₅₀ (5 sizes × 3 geometries = 15 clusters)
- **Sites:** bridge, hollow (3-fold), and atop positions
- **Method:** PBE + D3(BJ) dispersion, 350 eV plane-wave cutoff, Γ-point only
- **Reference:** Computational hydrogen electrode (CHE) with 0.24 eV ZPE correction
- **H* geometry:** MACE-MP-0 relaxed (cluster atoms frozen), DFT single-point

Each `results_CuN_XX.json` file contains per-site dictionaries with:

```json
{
  "site_id": "site_000_bridge",
  "site_type": "bridge",
  "dgh_gpaw_eV": -0.142,
  "dgh_mace_eV": -0.089,
  "dgh_chgnet_eV": 0.317,
  "dgh_tensornet_eV": -0.163,
  "dgh_mace_ft_N10_seed0_eV": -0.118,
  ...
}
```

**Full dataset with wavefunction files and model checkpoints:** [Zenodo DOI: ZENODO_DOI_PLACEHOLDER]

---

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cu-nanocluster-benchmark

# Verify GPU is available (required for MACE and CHGNet)
python3 -c "import torch; print(torch.cuda.is_available())"
```

**System requirements:**
- Python 3.11
- CUDA-capable GPU (tested: NVIDIA RTX 4060, 8 GB VRAM)
- ~16 GB RAM for DFT calculations
- GPAW requires a separate installation via conda-forge (included in environment.yml)

---

## Reproducing the Benchmark

```bash
# Step 1: Generate and relax Cu nanoclusters
python3 scripts/01_generate_clusters.py
python3 scripts/02_mace_relax.py

# Step 2: Enumerate adsorption sites
python3 scripts/03_adsorption_sites.py
python3 scripts/03b_atop_sites.py

# Step 3: Run GPAW DFT (parallelised, checkpointed — ~80 CPU-hours total)
python3 scripts/04a_gpaw_prep.py
python3 scripts/04b_gpaw_launch.py --sizes 10 20 30 40 50 --ncores 4

# Step 4: Zero-shot MLIP evaluation
python3 scripts/05_chgnet_dgh.py
python3 scripts/06_tensornet_dgh.py

# Step 5: Analysis and figures
python3 scripts/07_compare_results.py
python3 scripts/08_publication_figures.py
python3 scripts/10_rank_order_analysis.py

# Step 6: Fine-tuning learning curves
python3 scripts/09b_mace_ft_extended.py   # ~6 hr GPU
python3 scripts/09c_reconstruct_ext_lc.py
python3 scripts/19_regenerate_lc_figures.py
```

To skip the DFT step entirely, use the pre-computed data in `data/dft_reference/` and start from Step 4.

---

## Citation

If you use this dataset or benchmark in your work, please cite:

```bibtex
@article{tavana2026cu_nanocluster_benchmark,
  title   = {Benchmarking Foundation ML Interatomic Potentials for HER Activity
             Screening of Copper Nanoclusters},
  author  = {Tavana, Jalal},
  journal = {ACS Catalysis},
  year    = {2026},
  note    = {arXiv:ARXIV_ID_PLACEHOLDER}
}
```

Dataset citation:
```bibtex
@dataset{tavana2026cu_nanocluster_dataset,
  author    = {Tavana, Jalal},
  title     = {Cu Nanocluster MLIP Benchmark Dataset},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {ZENODO_DOI_PLACEHOLDER}
}
```

---

## License

Code: [MIT License](LICENSE)
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Contact

Jalal Tavana — jalal.tavana@pm.me
