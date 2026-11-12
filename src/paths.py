"""Single source of truth for filesystem locations used by every script."""
import os

# Outputs (processed data, results, figures, tables) go under the release root.
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Raw datasets are read in place and never copied or modified. Set
# ARC_DATA_DIR to the folder holding UNSW_NB15/ and CIC-IDS2017/.
RAW_ROOT = os.environ.get("ARC_DATA_DIR", os.path.join(PROJECT, "data_raw"))
UNSW_RAW_DIR = os.path.join(RAW_ROOT, "UNSW_NB15")
CIC_RAW_DIR = os.path.join(RAW_ROOT, "CIC-IDS2017")
# NF-CSE-CIC-IDS2018-v3 (Luay et al., 2025): the authors' original BagIt
# archive from UQ RDM (eSpace UQ:ece9b83), read directly from the zip.
NF_ZIP = os.environ.get("ARC_NF_ZIP", os.path.join(PROJECT, "data_raw", "NF-CICIDS2018-v3",
                                                   "NF-CICIDS2018-v3.zip"))

DATA_DIR = os.path.join(PROJECT, "data_processed")
RESULTS_RAW = os.path.join(PROJECT, "results", "raw")
RESULTS_PROC = os.path.join(PROJECT, "results", "processed")
RESULTS_STATS = os.path.join(PROJECT, "results", "statistics")
RESULTS_ABL = os.path.join(PROJECT, "results", "ablation")
CKPT_DIR = os.path.join(PROJECT, "checkpoints")
LOG_DIR = os.path.join(PROJECT, "logs")
FIG_DIR = os.path.join(PROJECT, "figures", "plots")
TAB_DIR = os.path.join(PROJECT, "tables")

for _d in (DATA_DIR, RESULTS_RAW, RESULTS_PROC, RESULTS_STATS, RESULTS_ABL,
           CKPT_DIR, LOG_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)
