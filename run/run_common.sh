#!/bin/bash
# model comparison with a COMMON fixed noise model and a profile weight of 6 effective points (lambda_profile = 1), no datum, warm starts, history started at 42 C
cd "$(dirname "$0")/.."
# activate the conda environment first (see environment.yml)
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ERR=1 VHR_SCALE=100 K=24 NTRAP=24 N_FINE=24 NEUTRAL=1 MAAD=1 B_OVER_A=2 LOGD0_HI=4.4
export NEFF_PROF=6 LAMFIX=tl=11.6,maad=1,lab=3.5,itl=3.5,prof=1 TSTART=42
A0F=gcd_params_irra024_joint_maad_ERR_nN_vs100.json; GKF=gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json
IMTS_MODEL=a0 python lm_inv.py $A0F lm_lmA00.json lmA0c 3 14 0 > log_lmA0c.txt 2>&1
IMTS_MODEL=gok python lm_inv.py $GKF lm_lmGOK0.json lmGOKc 3 14 0 > log_lmGOKc.txt 2>&1
touch COMMON_INDEP_DONE.marker
