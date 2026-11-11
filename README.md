# IMTS for Na-feldspar thermoluminescence paleothermometry

Code and derived inputs for the paper on the interactive multi-trap system (IMTS) for TL paleothermometry of Na-feldspar sample ANA-05 (Climate of the Past, in preparation). The IMTS (24 traps, charge neutral, single retrapping and recombination coefficients with B/A = 2) is compared with a general-order kinetic model (GOK) and with independent first-order traps (A = 0).

Status: unpublished work, private repository. Code under the MIT licence (see `LICENSE`); the data files are provided for reproducing the paper.

## Layout

| Path | Content |
|---|---|
| `fwd.py` | forward model of all data sets (TL ramps, MAAD growth, laboratory fading, ITL, natural profile) and the parameter vector |
| `lm_inv.py` | regularised inversion (Levenberg-Marquardt, Laplace posterior); usage in its docstring |
| `retrain_6beta_irr/` | IMTS ODE (irradiation, preheat, read-out) and the surrogate weights it needs at import |
| `indep.py`, `a0_model.py`, `gok_history.py`, `light_models.py` | independent-trap (A = 0) and GOK models |
| `imts_step.py`, `inv_joint.py`, `cap_tests.py` | geological history (holds, tunnelling fading, natural profile) |
| `tscan.py`, `tscan_split.py`, `eval_subset.py`, `dump_fits.py` | constant-temperature scans, window split, evaluation of stored fits |
| `synth.py`, `synprof.py` | synthetic data with known temperature and their profile curves |
| `run/` | the shell scripts that produced the results (common-noise fits, slow-rate subset, synthetic tests) |
| `lm_*.json`, `fits_lm*.pkl` | stored fits (`c` all rates, `s` slow rates) and their evaluations |
| `tl_data_ana05cups_*.txt`, `maad_windows.json`, `maad_thermometers.json`, `itl_data.pkl`, `fading_cycles.json`, `noise_model_ana05.json` | measured data in processed form (no raw instrument files) |
| `paper_figs/` | one script per paper figure, a common style (`style.py`, Times font, inside ticks, panel labels) and `tests_fwd_gpu/` for the synthetic temperature-history tests |

## Running

```
conda env create -f environment.yml && conda activate imts_jax
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 ERR=1 VHR_SCALE=100 K=24 NTRAP=24 N_FINE=24 NEUTRAL=1 MAAD=1 B_OVER_A=2 LOGD0_HI=4.4
bash run/run_slow.sh            # fits of the three models on the slow rates (needs hours per model)
```
All scripts must be run from the repository root. The environment variables above are read at import time and are set in every `run/*.sh`.

Figures (run the first three from the repository root, the last from `paper_figs/tests_fwd_gpu`)

```
PYTHONPATH=. python paper_figs/make_forward3_slow.py
python paper_figs/make_compare3_slow.py
python paper_figs/make_synthetic_tests.py
python paper_figs/make_maad_window_growth.py
cd paper_figs/tests_fwd_gpu && python example_sens_med.py && python epica_sens_med.py && python elev_post_med.py && python osc_post.py
```
Outputs go to `paper_figs/out/` (not tracked).

## Known gaps

- `paper_figs/tests_fwd_gpu/` contains the scripts that generate the grids of the synthetic temperature-history tests (`example_*.py`, `test*.py`, `osc_*.py`, JAX). They run from `paper_figs/tests_fwd_gpu` but call the forward model, which reads its data with paths relative to the repository root, so they have not been re-run from this repository yet; the plotting scripts (`*_med.py`, `osc_post.py`) have been re-run and reproduce the paper figures. Proxy records in `proxy/` are public (NGRIP d18O, GICC05 timescale, EPICA Dome C EDC3 temperature); please add the full citations before release.

- The synthetic 10 degC test is run (`run/run_synth10.sh`) but not yet in `make_synthetic_tests.py` / `synprof.py`.
- Raw instrument files (BIN) are not included.
