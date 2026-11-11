#!/bin/bash
# synthetic-data benchmark: truth = IMTS (lmBc kinetics) at a known present temperature; fit with IMTS (well-specified), then GOK and A=0 (wrong models). Common noise model, no datum, cold start (reference kinetics, T start 42).
cd "$(dirname "$0")/.."
# activate the conda environment first (see environment.yml)
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ERR=1 VHR_SCALE=100 K=24 NTRAP=24 N_FINE=24 NEUTRAL=1 MAAD=1 B_OVER_A=2 LOGD0_HI=4.4
IM=start_joint1_final.json; A0F=gcd_params_irra024_joint_maad_ERR_nN_vs100.json; GKF=gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json
while [ ! -f OVERNIGHT_DONE.marker ]; do sleep 120; done
export NEFF_PROF=6 LAMFIX=tl=11.6,maad=1,lab=3.5,itl=3.5,prof=1 TSTART=42
for TT in 50 30; do
  [ -f synth_T$TT.pkl ] || IMTS_MODEL=imts IMTS_PARAMS=$IM python synth.py gen lm_lmBc.json $TT 1 1 synth_T$TT.pkl > log_synth_gen_$TT.txt 2>&1
  SYNTH_DATA=synth_T$TT.pkl IMTS_MODEL=imts IMTS_PARAMS=$IM python lm_inv.py $IM $IM synI_T$TT 8 14 0 > log_synI_T$TT.txt 2>&1
  SYNTH_DATA=synth_T$TT.pkl IMTS_MODEL=gok  IMTS_PARAMS=$GKF python lm_inv.py $GKF $GKF synG_T$TT 4 14 0 > log_synG_T$TT.txt 2>&1
  SYNTH_DATA=synth_T$TT.pkl IMTS_MODEL=a0   IMTS_PARAMS=$A0F python lm_inv.py $A0F $A0F synA_T$TT 4 14 0 > log_synA_T$TT.txt 2>&1
  touch SYNTH_T${TT}_DONE.marker
done
touch SYNTH_DONE.marker
