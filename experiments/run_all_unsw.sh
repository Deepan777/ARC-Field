#!/bin/bash
# Sequential main runs (RAM limits parallelism). Logs in logs/main/.
cd "$(dirname "$0")"
python run_unsw.py --protocol chrono --seeds 0 1 2 3 4 > ../../logs/main/unsw_chrono_A.log 2>&1
python run_unsw.py --protocol hostdisjoint --seeds 0 1 2 3 4 > ../../logs/main/unsw_hostdisjoint_A.log 2>&1
python run_unsw.py --protocol random --seeds 0 1 2 > ../../logs/main/unsw_random_A.log 2>&1
echo ALLDONE > ../../logs/main/run_all_unsw.done
