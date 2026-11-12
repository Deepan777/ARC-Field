# ARC-Field: code and results

Code for *ARC-Field: Behavioural Endpoint Roles for Benign-Only Intrusion
Detection on Encrypted-Traffic Metadata Under Deployment Conditions*. Every
reported result is produced by the scripts below; the manuscript reads the
numbers from generated LaTeX macros.

## Relation to GraphINR-IDS

This work extends the authors' earlier study: P. Deepanramkumar and A. Helen
Sharmila, "GraphINR-IDS: Graph-topology-aware implicit neural reconstruction
for zero-day intrusion detection in encrypted network traffic," *J. King Saud
Univ. – Comput. Inf. Sci.*, vol. 38, no. 8, Art. no. 916, 2026,
doi:[10.1007/s44443-026-01260-x](https://doi.org/10.1007/s44443-026-01260-x).
The eleven baselines in `third_party/baselines.py` are its implementations,
reused unchanged. GraphINR-IDS itself is one of the baselines of this paper:
its model is in `third_party/graphinr_ids/` (unchanged), its spectral and
absolute-time coordinates, min-max preprocessing, published configuration and
training loop are in `src/graphinr_ids_harness.py`, and `src/graphinr_baseline.py`
runs it on every protocol with its own scorer. Its spectral and time
coordinates also serve as comparators in the ablation and churn studies. The
observability tiers of the feature sets follow its audit.

## Layout

| Folder | Contents |
|---|---|
| `src/` | ARC-Field: causal role coordinates (`context.py`), the masked sinusoidal field (`arcfield.py`), robust scaling, the ARC recalibration and comparison policies (`calibration.py`), metrics, data loaders, extra baselines (KitNET, MCM, DIF, Anomal-E), the GraphINR-IDS harness and runner (`graphinr_ids_harness.py`, `graphinr_baseline.py`), and the alternative endpoint encodings, Kitsune's AfterImage and ReFeX/RolX roles (`context_alt.py`) |
| `third_party/` | The eleven standard one-class baselines (IF, LOF, OC-SVM, dense/LSTM/Transformer AE, Deep SVDD, DAGMM, MemAE, USAD, DDPM) and the GraphINR-IDS model (`graphinr_ids/`) |
| `experiments/` | One script per study: main protocols (`run_unsw.py`), model ablation, calibration study, poisoning, churn, CIC-IDS2017, NF-CICIDS2018-v3 (`run_nf.py`, Protocol V), HIKARI-2021 (`run_hikari.py`, Protocol VI), the comparison of endpoint encodings (`run_context_baselines.py`), the role-anomaly extension (`run_role_anomaly.py`), complexity |
| `analysis/` | Aggregation, paired tests with Holm correction, Friedman–Nemenyi, stationarity/attribution, per-family analysis, tables, macros, figures |
| `tools/` | Corpus construction (`profile_unsw.py`), the causality check, development-protocol selection scripts (`dev_*.py`), `smoke_test.py`, and `audit/`: the development/test overlap count (`issue1_overlap.py`, Protocols I and III against Protocol D), the spectral-scaling diagnostic (`issue2_spectral.py`) and a GraphINR-IDS reproduction check |

## Setup

Python 3.11. Tested with torch 2.13 (CUDA 12.6), numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.8.0, pandas 3.0.3, pyarrow 24.0.0, matplotlib 3.10.8.
A GPU is optional; every job falls back to the CPU.

Point `ARC_DATA_DIR` to a folder containing the original CSV releases:

```
$ARC_DATA_DIR/UNSW_NB15/UNSW-NB15_{1..4}.csv, NUSW-NB15_features.csv
$ARC_DATA_DIR/CIC-IDS2017/<MachineLearningCVE day files>.csv
```

For Protocol V, download the NF-CICIDS2018-v3 archive (Luay et al., IEEE Access
2026; UQ eSpace record UQ:ece9b83, open access) and set `ARC_NF_ZIP` to the
zip file, or place it at `data_raw/NF-CICIDS2018-v3/NF-CICIDS2018-v3.zip`. The
loader reads the CSV directly from the zip. The expected SHA-1 of the CSV is
`ddcad6d826b73a128a447a0189137efd6c8f70c9`.

For Protocol VI, download `ALLFLOWMETER_HIKARI2021.csv.zip` from the HIKARI-2021
Zenodo record 5199540 (Ferriyan et al., Applied Sciences 2021; CC BY 4.0;
MD5 `d7d9e277fe4a66cb00764d7f91a810dd`) into `data_raw/HIKARI-2021/`, or set
`ARC_HIKARI_ZIP`. Build its cache with `python -I tools/build_hikari.py`.

The comparison of endpoint encodings needs the AfterImage and ReFeX features of
both stream corpora: `python tools/verify_ctxalt.py` checks the encoders
against brute-force references, and `python tools/build_ctxalt.py` computes
them (about 3 GB under `data_processed/ctxalt/`).

## Reproduce

```bash
python tools/smoke_test.py          # synthetic, CPU, about a minute
python tools/profile_unsw.py        # builds data_processed/unsw_full.parquet
python tools/verify_causality.py    # checks that the context is causal
bash run_pipeline.sh                # all experiments, then tables and figures
```

`run_pipeline.sh` is resumable. Every run writes its own result file
atomically, each stage records a `.done` marker in `logs/pipeline/`, and
re-running the script skips completed work. Each (seed, method) job runs in its
own process, and the last retry falls back to the CPU. The full pipeline takes
roughly two days on one laptop GPU (RTX-class, 6 GB).

Hyper-parameters are frozen in `experiments/common.py` (`ARC_HP`) and
`src/calibration.py` (`CalConfig`). They were selected on the development
protocol only (`tools/dev_*.py`), using its attack labels. That protocol
streams the first two hours of UNSW-NB15 campaign 1, which contain every
campaign-1 attack; Protocols I and III evaluate on those attacks and are
therefore reported as non-independent analyses
(`tools/audit/issue1_overlap.py` counts the overlap). No baseline was tuned on
the development protocol.

## Outputs

- `results/raw/`: one JSON file per run
- `results/statistics/`: paired tests and Friedman–Nemenyi
- `manuscript/tables/numbers.tex`: all reported numbers as macros
- `manuscript/figures/`: all plots

## Results included in this repository

This repository holds the code together with:

- `results/raw/`: the result file of every run, as JSON
- `results/processed/`, `results/statistics/`, `results/ablation/`,
  `results/context_baselines/`, `results/role_anomaly/`: derived results
- `logs/dev/`: the development-protocol selection logs
- `data_processed/*_profile.json`: corpus profiles
- `reproducibility/`: the causality check and the pinned environment

From these files alone, or after building the two corpora noted below:

```bash
cd analysis
python aggregate.py     # results/statistics, from results/raw
python make_tables.py   # results tables and manuscript/tables/numbers.tex
python figures.py       # data figures
```

`make_tables.py` also needs `data_processed/cic_arcfield_corpus.npz` and
`data_processed/hikari2021_corpus.npz`, built from the public datasets (see
Setup), and it reads the CIC-IDS2017 file headers through `ARC_DATA_DIR`. The
datasets are not redistributed. Run this way, from a clean copy of this repository,
these scripts reproduced every macro and generated table of the manuscript
exactly.

The cached detector outputs (`results/raw/scores/` and other `*.npz` files) are
not in this repository. `analysis_family_matched.py`,
`analysis_stationarity_attribution.py`, `analysis_tail_transfer.py`,
`analysis_ttl_collateral.py` and `analysis_context_drift.py` read them; their
outputs are included under `results/processed/`. The cached outputs are available
from the corresponding author on request, and `run_pipeline.sh` re-creates them.
