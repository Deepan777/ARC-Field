#!/bin/bash
# ============================================================================
# ARC-Field: complete, RESUMABLE experiment pipeline.
# Safe to re-run at any time: every run writes its own result file atomically
# and every stage skips work that is already done, so an interruption (sleep,
# shutdown, crash) costs at most the run that was in progress.
#   Resume:  bash run_pipeline.sh
#   Progress: logs/pipeline/progress.log
# ============================================================================
cd "$(dirname "$0")/experiments"
L=../logs/pipeline; mkdir -p "$L"
LOCK="$L/pipeline.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "pipeline already running (pid $(cat "$LOCK"))"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
stage () {  # stage <name> <command...>
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then echo "$(date '+%F %T') skip $name (done)" >> "$L/progress.log"; return; fi
  echo "$(date '+%F %T') start $name" >> "$L/progress.log"
  if "$@" >> "$L/$name.log" 2>&1; then
    touch "$L/$name.done"; echo "$(date '+%F %T') done  $name" >> "$L/progress.log"
  else
    echo "$(date '+%F %T') FAIL  $name (exit $?) - see $L/$name.log" >> "$L/progress.log"
  fi
}
stage main_chrono_v2   python run_unsw.py --protocol chrono --seeds 0 1 2 3 4
stage main_host_v2     python run_unsw.py --protocol hostdisjoint --seeds 0 1 2 3 4
stage main_random_v2   python run_unsw.py --protocol random --seeds 0 1 2
stage ablation_chrono  python run_ablation_model.py --protocol chrono --seeds 0 1 2 3 4
stage ablation_host    python run_ablation_model.py --protocol hostdisjoint --seeds 0 1 2 3 4
stage calib_chrono     python run_calibration_study.py --protocol chrono
stage calib_host       python run_calibration_study.py --protocol hostdisjoint
stage poisoning        python run_poisoning.py
stage churn            python run_churn.py --seeds 0 1 2
for fs in A_noTTL AB ABC; do
  stage featureset_$fs python run_unsw.py --protocol chrono --featureset $fs --seeds 0 1 2 --no-save-scores \
      --methods ARC-Field MemAE "Dense Autoencoder" KitNET MCM "Deep SVDD" DIF
done
stage cic              python run_cic.py --seeds 0 1 2
# Protocol V: NF-CSE-CIC-IDS2018-v3 (addresses + timestamps; cross-day)
stage nf_main          python run_nf.py --featureset A --seeds 0 1 2
for fs in A_noTTL AB; do
  stage nf_featureset_$fs python run_nf.py --featureset $fs --seeds 0 1 2 --no-save-scores \
      --methods ARC-Field MemAE "Dense Autoencoder" KitNET MCM "Deep SVDD" DIF
done
stage complexity       python run_complexity.py
# Context-baseline study: role coordinates vs Kitsune's AfterImage and RolX roles
stage ctxalt_build     python ../tools/build_ctxalt.py --corpus unsw nf
stage ctx_chrono       python run_context_baselines.py --protocol chrono
stage ctx_host         python run_context_baselines.py --protocol hostdisjoint
stage ctx_nf           python run_context_baselines.py --protocol nf
# Protocol VI: HIKARI-2021, genuinely encrypted (HTTPS) attacks, flow level
stage hikari_build     python -I ../tools/build_hikari.py
stage hikari_shift     python run_hikari.py --setting shift --seeds 0 1 2
stage hikari_random    python run_hikari.py --setting random --seeds 0 1 2
# Role-anomaly extension (pre-registered before its runs; decisions_log item 26)
for p in dev chrono hostdisjoint nf; do
  stage role_anom_$p   python run_role_anomaly.py --protocol $p
done
# GraphINR-IDS as published, on every protocol (decisions_log item 30)
stage graphinr_random  python run_unsw.py --protocol random --seeds 0 1 2 --methods GraphINR-IDS
stage graphinr_chrono  python run_unsw.py --protocol chrono --seeds 0 1 2 3 4 --methods GraphINR-IDS
stage graphinr_host    python run_unsw.py --protocol hostdisjoint --seeds 0 1 2 3 4 --methods GraphINR-IDS
stage graphinr_cic     python run_cic.py --seeds 0 1 2 --methods GraphINR-IDS
stage graphinr_nf      python run_nf.py --featureset A --seeds 0 1 2 --methods GraphINR-IDS
stage graphinr_hikari_shift  python run_hikari.py --setting shift --seeds 0 1 2 --methods GraphINR-IDS
stage graphinr_hikari_random python run_hikari.py --setting random --seeds 0 1 2 --methods GraphINR-IDS
# development/test overlap audit (Protocols I and III vs Protocol D)
stage dev_overlap      python ../tools/audit/issue1_overlap.py
cd ../analysis
stage analysis_stationarity python analysis_stationarity_attribution.py
stage analysis_family  python analysis_family_matched.py
stage analysis_ttl     python analysis_ttl_collateral.py
stage analysis_tail    python analysis_tail_transfer.py
python analysis_context_baselines.py >> "$L/analysis_context.log" 2>&1
stage analysis_drift   python analysis_context_drift.py
# aggregation/tables/figures are cheap and always refreshed
python aggregate.py   >> "$L/aggregate.log" 2>&1
python make_tables.py >> "$L/tables.log" 2>&1
python figures.py     >> "$L/figures.log" 2>&1
echo "$(date '+%F %T') PIPELINE COMPLETE" >> "$L/progress.log"
touch "$L/pipeline.complete"
