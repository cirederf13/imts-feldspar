#!/bin/bash
cd "$(dirname "$0")/.."
# activate the conda environment first (see environment.yml)
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ERR=1 VHR_SCALE=100 K=24 NTRAP=24 N_FINE=24 NEUTRAL=1 MAAD=1 B_OVER_A=2 LOGD0_HI=4.4 IMTS_PARAMS=start_joint1_final.json
export NEFF_PROF=6 LAMFIX=tl=11.6,maad=1,lab=3.5,itl=3.5,prof=1 TSTART=42
python lm_inv.py start_joint1_final.json lm_lmB.json lmBc 5 12 0 > log_lmBc.txt 2>&1
touch COMMON_IMTS_DONE.marker
