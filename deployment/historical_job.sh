#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LANCE_CPU_THREADS=2 LANCE_IO_THREADS=2 RAYON_NUM_THREADS=2 MALLOC_ARENA_MAX=2
export HISTORICAL_SUBCOMPARTMENT_WORK=${HISTORICAL_SUBCOMPARTMENT_WORK:-/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/historical117_subcompartment_six_class_v1}
export SUBCOMPARTMENT_REPO="$HISTORICAL_SUBCOMPARTMENT_WORK/code"
cd "$SUBCOMPARTMENT_REPO"
worker_pid=''
finish_worker() {
  if [[ -n "$worker_pid" ]]; then kill -TERM "$worker_pid" 2>/dev/null || true; wait "$worker_pid" 2>/dev/null || true; fi
}
resume_worker() {
  finish_worker
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap resume_worker USR1
trap 'finish_worker; exit 143' TERM
/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python -u "$SUBCOMPARTMENT_REPO/historical.py" "$@" &
worker_pid=$!
wait "$worker_pid"
