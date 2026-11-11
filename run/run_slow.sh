#!/bin/bash
# slow-rate subset (0.1-1 C/s), common fixed noise, no datum, all three models, warm start from the all-rate common-noise solutions
cd "$(dirname "$0")/.."
# activate the conda environment first (see environment.yml)
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ERR=1 VHR_SCALE=100 K=24 NTRAP=24 N_FINE=24 NEUTRAL=1 MAAD=1 B_OVER_A=2 LOGD0_HI=4.4
export NEFF_PROF=6 LAMFIX=tl=11.6,maad=1,lab=3.5,itl=3.5,prof=1 TL_RATES=0,1,2,3
IM=start_joint1_final.json; A0F=gcd_params_irra024_joint_maad_ERR_nN_vs100.json; GKF=gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json
(IMTS_MODEL=imts IMTS_PARAMS=$IM python lm_inv.py $IM lm_lmBc.json lmBs 4 12 0 > log_lmBs.txt 2>&1) &
(IMTS_MODEL=gok IMTS_PARAMS=$GKF python lm_inv.py $GKF lm_lmGOKc.json lmGOKs 4 12 0 > log_lmGOKs.txt 2>&1) &
(IMTS_MODEL=a0 IMTS_PARAMS=$A0F python lm_inv.py $A0F lm_lmA0c.json lmA0s 4 12 0 > log_lmA0s.txt 2>&1) &
wait; touch SLOW_DONE.marker
