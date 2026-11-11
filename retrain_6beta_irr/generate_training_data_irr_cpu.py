"""
IMTS training data WITH the laboratory irradiation simulated in the ODE, then preheat, then read-out (shared N(E) across all stages).

Protocol per sample (all traps empty at the start, as after a TL to 450 C):
  stage 0  irradiation at T_IRR=293.15 K for t = D/Ddot_lab (Ddot_lab = 0.1566 Gy/s, the cups' beta source), full IMTS ODE with
           radiogenic generation G = (Ddot_lab/D0) * sum(N_i) into the conduction band (chen_pagonis_vs_gok.pdf Sec. 3.1), thermal release included
  stage 1  n_c reset to 0 (cooling/delay ignored), ramp T0 -> PH at beta  (same as the PH version)
  stage 2  n_c reset to 0, ramp T0 -> T_max at beta; curve = B*n_c*N_r, peak-normalised, plus log10 peak
The capture coefficient may depend on the trap energy:  A_i = A * 10^(kappa*(E_i - 1.5)).   (A=1e-8, B=1e-8, N_r=1e10 fixed)
Inputs (16): [slope, intercept, delta_ctrl_1..5, logN_ctrl_1..5, PH_celsius, log10 D (Gy), log10 D0 (Gy), kappa]
usage: python generate_training_data_irr_cpu.py N_samples n_jobs seed0 outfile
"""
import os, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed

k_B   = 8.617333262e-5
T0    = 295.15
T_IRR = 293.15
T_max = 745.0
A_val = 1e-8; B, N_r = A_val * float(os.environ.get('B_OVER_A', '1.0')), 1e10          # B_OVER_A = A_m / A_n (default 1)
NEUTRAL = os.environ.get('NEUTRAL', '0') == '1'; M0 = 1e10                      # NEUTRAL=1: charge-neutral recombination centres, m = M0 + sum(n) + n_c (M0 set per call)
DOT   = 0.1566            # Gy/s, beta source of the cups (1629 s = 255 Gy; 1916 s = 300 Gy)

N_FINE = int(os.environ.get('N_FINE', 16))                      # number of traps on the energy grid (default 16)
E_lo, E_hi = 0.95, 2.05
E_GRID = np.linspace(E_lo, E_hi, N_FINE)
N_CTRL = 5
E_CTRL = np.linspace(E_lo, E_hi, N_CTRL)

slope_lo, slope_hi         = -1.0, 6.0
intercept_lo, intercept_hi = 3.0, 17.0
delta_ctrl_bound           = 1.0
logN_lo, logN_hi           = 8.0, 11.0
logs_safe_lo, logs_safe_hi = 9.0, 17.0
PH_lo, PH_hi               = 20.0, 320.0
logD_lo, logD_hi           = 1.5, 4.3         # 32 Gy ... 20 kGy (MAAD: 51 ... 10200 Gy; VHR/ITL: 255-300 Gy)
logD0_lo, logD0_hi         = 2.3, 5.0
kappa_lo, kappa_hi         = -1.0, 2.0        # log10 of capture coefficient per eV, relative to E = 1.5 eV
FILL_MIN                   = 2e-3             # reject if the irradiation leaves less than this fraction of total capacity filled
DOD0_MIN                   = 3e-3             # reject draws with D/D0 below this (cheap, before any ODE)
Q_MIN = 1e-3                                  # reject samples where the preheat empties the traps

N_T   = 300
T_out = np.linspace(T0, T_max, N_T)
BETAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

def spline_eval(ctrl_vals):
    return CubicSpline(E_CTRL, ctrl_vals, bc_type='natural')(E_GRID)

def make_rhs(E, s, N, A, beta=None, gen=0.0, T_const=None):
    """full IMTS RHS; ramp at rate beta from T0, or constant temperature T_const with radiogenic generation gen (charge/s)"""
    def rhs(t, y):
        n, n_c = y[:-1], y[-1]
        T = T_const if T_const is not None else T0 + beta * t
        detr = s * n * np.exp(-E / (k_B * T))
        retr = A * n_c * (N - n)
        loss = n_c * (np.sum(A * (N - n)) + B * ((M0 + np.sum(n) + n_c) if NEUTRAL else N_r))
        return np.concatenate([-detr + retr, [np.sum(detr) + gen - loss]])
    return rhs

def irradiate(E, s, N, A, D, D0):
    t_irr = D / DOT; gen = (DOT / D0) * np.sum(N)
    sol = solve_ivp(make_rhs(E, s, N, A, gen=gen, T_const=T_IRR), [0, t_irr], np.zeros(len(N) + 1), method='LSODA',
                    rtol=1e-6, atol=1e-8, first_step=1e-3)
    if not sol.success: return None
    return np.clip(sol.y[:-1, -1], 0.0, N)

def run_imts_irr(E, s, N, A, n_irr, beta, T_PH):
    """returns (normalised curve, log10 peak, remaining-charge fraction after preheat) or None"""
    n_start = n_irr
    if T_PH > T0 + 1e-6:
        t1 = (T_PH - T0) / beta
        sol1 = solve_ivp(make_rhs(E, s, N, A, beta=beta), [0, t1], np.concatenate([n_irr, [0.0]]), method='LSODA',
                         max_step=0.5 / beta, rtol=1e-6, atol=1e-8)
        if not sol1.success: return None
        n_start = np.maximum(sol1.y[:-1, -1], 0.0)
    q_rem = np.sum(n_start) / np.sum(n_irr)
    t_max = (T_max - T0) / beta; t_out = (T_out - T0) / beta
    sol = solve_ivp(make_rhs(E, s, N, A, beta=beta), [0, t_max], np.concatenate([n_start, [0.0]]), method='LSODA',
                    dense_output=True, max_step=0.5 / beta, rtol=1e-6, atol=1e-8)
    if not sol.success: return None
    Y = sol.sol(t_out); I = B * Y[-1] * ((M0 + Y[:-1].sum(0) + Y[-1]) if NEUTRAL else N_r); mx = np.max(I)
    if mx <= 0: return None
    return I / mx, np.log10(mx), q_rem

def draw_params(rng):
    slope = rng.uniform(slope_lo, slope_hi)
    intercept = rng.uniform(intercept_lo, intercept_hi)
    delta = rng.uniform(-delta_ctrl_bound, delta_ctrl_bound, N_CTRL) * rng.uniform(0.15, 1.0)
    if rng.random() < 0.4:
        a, b = rng.uniform(logN_lo, logN_hi, 2)
        logN = np.clip(a + (b - a) * np.linspace(0, 1, N_CTRL) + rng.normal(0, 0.05, N_CTRL), logN_lo, logN_hi)
    else:
        logN = rng.uniform(logN_lo, logN_hi, N_CTRL)
    PH = rng.uniform(PH_lo, PH_hi)
    logD = rng.uniform(logD_lo, logD_hi); logD0 = rng.uniform(logD0_lo, logD0_hi); kappa = rng.uniform(kappa_lo, kappa_hi)
    return slope, intercept, delta, logN, PH, logD, logD0, kappa

ST_VALID, ST_ODE_FAIL, ST_PREHEAT_EMPTY, ST_LOGS_RANGE, ST_NO_CHARGE = 0, 1, 2, 3, 4
STATUS_NAMES = {0: 'valid', 1: 'ODE solve failed (see beta_fail; -2 = irradiation)', 2: 'preheat emptied traps (min q_rem < Q_MIN)',
                3: 'log s outside safe range (ODE not run)', 4: 'D/D0 too small or irradiation filled < FILL_MIN'}
N_IN = 16

def generate_one(seed):
    """one record per drawn model (up to 50 per seed, stops at the first valid one):
    (params16, status, beta_fail_index, min_qrem, logpeak[6], qrem[6], draw_index, seconds, fill_total)"""
    rng = np.random.default_rng(seed); records = []
    for d in range(50):
        slope, intercept, delta, logN, PH, logD, logD0, kappa = draw_params(rng)
        pv = np.concatenate([[slope, intercept], delta, logN, [PH, logD, logD0, kappa]])
        nan6 = np.full(6, np.nan)
        log_s = slope * E_GRID + intercept + spline_eval(delta)
        if np.any(log_s < logs_safe_lo) or np.any(log_s > logs_safe_hi):
            records.append((pv, ST_LOGS_RANGE, -1, np.nan, nan6, nan6, d, 0.0, np.nan)); continue
        if logD - logD0 < np.log10(DOD0_MIN):
            records.append((pv, ST_NO_CHARGE, -1, np.nan, nan6, nan6, d, 0.0, np.nan)); continue
        s_vals, N_vals = 10.0 ** log_s, 10.0 ** spline_eval(logN)
        A_vec = A_val * 10.0 ** (kappa * (E_GRID - 1.5)); T_PH = PH + 273.15
        t0 = time.time()
        n_irr = irradiate(E_GRID, s_vals, N_vals, A_vec, 10.0 ** logD, 10.0 ** logD0)
        if n_irr is None:
            records.append((pv, ST_ODE_FAIL, -2, np.nan, nan6, nan6, d, time.time() - t0, np.nan)); continue
        fill = np.sum(n_irr) / np.sum(N_vals)
        if fill < FILL_MIN:
            records.append((pv, ST_NO_CHARGE, -1, np.nan, nan6, nan6, d, time.time() - t0, fill)); continue
        curves, logpk, qs = [], np.full(6, np.nan), np.full(6, np.nan); fail = -1
        for ib, beta in enumerate(BETAS):
            r = run_imts_irr(E_GRID, s_vals, N_vals, A_vec, n_irr, beta, T_PH)
            if r is None: fail = ib; break
            curves.append(r[0]); logpk[ib] = r[1]; qs[ib] = r[2]
        dt = time.time() - t0
        if fail >= 0:
            records.append((pv, ST_ODE_FAIL, fail, np.nanmin(qs) if np.any(np.isfinite(qs)) else np.nan, logpk, qs, d, dt, fill)); continue
        if np.min(qs) < Q_MIN:
            records.append((pv, ST_PREHEAT_EMPTY, -1, np.min(qs), logpk, qs, d, dt, fill)); continue
        records.append((pv, ST_VALID, -1, np.min(qs), logpk, qs, d, dt, fill))
        return records, np.array(curves, dtype=np.float32)
    return records, None

if __name__ == "__main__":
    import json, hashlib, platform
    N_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_jobs    = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed0     = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out_path  = sys.argv[4] if len(sys.argv) > 4 else 'training_data_irr_test.npz'
    print(f"Generating >= {N_samples} valid samples (16 inputs incl. PH, log dose, log D0, kappa), betas={BETAS}", flush=True)
    seeds = seed0 + np.arange(int(N_samples * 1.35)); t_start = time.time()
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(delayed(generate_one)(int(s)) for s in seeds)
    P, ST, BF, QM, LP, QR, DI, SEC, FL, SD, CUR_IDX, CUR = [], [], [], [], [], [], [], [], [], [], [], []
    for seed, (recs, cur) in zip(seeds, results):
        for r in recs:
            if r[1] == ST_VALID: CUR_IDX.append(len(P))
            P.append(r[0]); ST.append(r[1]); BF.append(r[2]); QM.append(r[3]); LP.append(r[4]); QR.append(r[5]); DI.append(r[6]); SEC.append(r[7]); FL.append(r[8]); SD.append(seed)
        if cur is not None: CUR.append(cur)
    CUR = np.array(CUR); ST = np.array(ST); CUR_IDX = np.array(CUR_IDX); nv = len(CUR_IDX)
    print(f"\nDrew {len(P)} models over {len(seeds)} seeds: " + ", ".join(f"{STATUS_NAMES[k]}: {(ST==k).sum()}" for k in STATUS_NAMES))
    kw = {f"curves_b{i}": CUR[:, i, :] for i in range(len(BETAS))}
    np.savez(out_path, params=np.array(P)[CUR_IDX], logpeak=np.array(LP)[CUR_IDX], qrem=np.array(QR)[CUR_IDX], fill=np.array(FL)[CUR_IDX],
             seed_valid=np.array(SD)[CUR_IDX], draw_index_valid=np.array(DI)[CUR_IDX],
             all_params=np.array(P), all_status=ST, all_beta_fail=np.array(BF), all_min_qrem=np.array(QM), all_logpeak=np.array(LP),
             all_qrem=np.array(QR), all_fill=np.array(FL), all_seed=np.array(SD), all_draw_index=np.array(DI), all_seconds=np.array(SEC),
             valid_index_in_all=CUR_IDX, betas=np.array(BETAS), T_out=T_out, E_grid=E_GRID, E_ctrl=E_CTRL, **kw)
    src = open(__file__, 'rb').read()
    manifest = dict(out_file=out_path, n_valid=int(nv), n_models_drawn=int(len(P)), n_seeds=int(len(seeds)), seed0=int(seed0),
        status_counts={STATUS_NAMES[k]: int((ST == k).sum()) for k in STATUS_NAMES}, script=os.path.basename(__file__),
        script_sha256=hashlib.sha256(src).hexdigest(), command=' '.join(sys.argv), host=platform.node(), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
        wall_seconds=round(time.time() - t_start),
        constants=dict(B=B, N_r=N_r, A=A_val, T0=T0, T_IRR=T_IRR, T_max=T_max, DOT_Gy_per_s=DOT, E_range=[E_lo, E_hi], n_traps=N_FINE, n_ctrl=N_CTRL,
                       slope=[slope_lo, slope_hi], intercept=[intercept_lo, intercept_hi], delta_bound=delta_ctrl_bound, logN=[logN_lo, logN_hi],
                       logs_safe=[logs_safe_lo, logs_safe_hi], PH=[PH_lo, PH_hi], logD=[logD_lo, logD_hi], logD0=[logD0_lo, logD0_hi],
                       kappa=[kappa_lo, kappa_hi], FILL_MIN=FILL_MIN, DOD0_MIN=DOD0_MIN, Q_MIN=Q_MIN, betas=BETAS,
                       solver='LSODA rtol=1e-6 atol=1e-8 (ramps max_step=0.5/beta; irradiation first_step=1e-3)',
                       protocol='traps empty -> irradiation at 293.15 K (full ODE, G=(Ddot/D0)*sumN, thermal release on) -> n_c reset -> ramp T0->PH at beta -> n_c reset -> ramp T0->T_max',
                       capture='A_i = A * 10^(kappa*(E_i-1.5))'))
    json.dump(manifest, open(out_path.replace('.npz', '_manifest.json'), 'w'), indent=1)
    print(f"Saved {nv} valid samples (+ all {len(P)} drawn models) to {out_path} in {time.time()-t_start:.0f}s; params {np.array(P)[CUR_IDX].shape}")
