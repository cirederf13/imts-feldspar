"""Regularised inversion by Levenberg-Marquardt on the stacked vector of whitened data residuals and prior residuals.
usage: python lm_inv.py ref.json start_inv_joint.json tag [n_jobs] [max_iter] [use_datum 1|0]
Noise: sigma_k * sqrt(lambda_k), lambda_k = max(1, mean residual^2) re-estimated every iteration (limited change); effective independent points per curve: TL 8, ITL 3 per condition, profile 3.
TL residuals: Student-t (nu=4) IRLS weights.  Priors: present T 47 +- 5 (optional), history near the present value (sigma_dev 2 C) and smooth (2 C), dE(Tm) <= 0.1 eV (second difference 0.05), eta 0.3 dex, dN 1 dex, weak D0/M0/Dnat, rho' law a ~ N(-5.5, 0.3), b ~ N(0, 0.3), log10 s soft-bounded to [10, 14]."""
import sys, os, json, time, numpy as np
f_ref, f_start, tag = sys.argv[1], sys.argv[2], sys.argv[3]; nj = int(sys.argv[4]) if len(sys.argv) > 4 else 11; MAXIT = int(sys.argv[5]) if len(sys.argv) > 5 else 12; USE_DATUM = (sys.argv[6] != "0") if len(sys.argv) > 6 else True
os.environ["IMTS_PARAMS"] = os.path.abspath(f_ref); import fwd, pgrid
fwd.init_main(); ref = pgrid.Ref(os.path.abspath(f_ref)); IDX = fwd.IDX; NX = fwd.NX; NXB = NX - (2 if fwd.QUENCH else 0)
if os.environ.get('KTB'): import ktb; ktb.apply(os.environ['KTB'])                      # KTB20-16 data terms (ktb.py)
if os.environ.get('SYNTH_DATA'): import synth; synth.apply(os.environ['SYNTH_DATA'])          # synthetic-data benchmark (synth.py)
SIGDEV = float(os.environ.get("SIGDEV", 2.0)); SIGSM = float(os.environ.get("SIGSM", 2.0)); SIGE = float(os.environ.get("SIGE", 0.10)); NU = 4.0
KAPPA = dict(tl=float(os.environ.get('NEFF_TL', 8.0)) / 268.0, itl=float(os.environ.get('NEFF_ITL', 3.0)) / 10.8, prof=float(os.environ.get('NEFF_PROF', 3.0)) / 15.0, maad=1.0, lab=1.0)      # effective independent points per curve (env NEFF_TL, NEFF_ITL)
def prior_resid(x):
    T = x[IDX["T"]]; r = []
    if USE_DATUM: r.append([(T[-1] - 47.0) / 5.0])
    r.append((T[:-1] - T[-1]) / SIGDEV); r.append((T[:-1] - T[1:]) / SIGSM)
    for nm, sd, sd2 in (("dE", SIGE, 0.05), ("eta", 0.3, 0.2), ("dN", 1.0, 1.0)):
        v = x[IDX[nm]]; r.append(v / sd); r.append((v[:-2] - 2 * v[1:-1] + v[2:]) / sd2)
    r.append([(x[IDX["logD0"]] - 3.5) / 1.0, (x[IDX["logM0"]] - 9.5) / 1.5])
    if NXB >= 29: bb = x[IDX["bexp"]]; r.append([max(0.0, 1.0 - bb) / 0.05, max(0.0, bb - 2.0) / 0.05, (bb - 1.5) / 1.0])
    if NXB == 30: aa = x[IDX["aexp"]]; r.append([max(0.0, 0.5 - aa) / 0.05, max(0.0, aa - 2.5) / 0.05, (aa - 1.0) / 1.0]); r.append((x[IDX["logDnat"]] - 1.0) / 2.0); r.append([(x[IDX["a"]] + 5.5) / 0.3, x[IDX["b"]] / 0.3])
    if os.environ.get('KTB'): r.append((x[IDX['logDnat']] - 2.5) / 1.5); r.append(np.maximum(0, 0.3 - x[IDX['logDnat']]) / 0.1); r.append(np.maximum(0, x[IDX['logDnat']] - 4.2) / 0.1)      # natural-dose nodes: weak prior (300 Gy, 1.5 dex) and soft bounds [0.3, 4.2] (the code clips beyond)
    if os.environ.get('KTB') or 'lab' in os.environ.get('DROP', '').split(','): r.append([(x[IDX['a']] + 5.5) / 0.3, x[IDX['b']] / 0.3])      # fading law is not fitted for KTB (rho' per window is given): keep a, b at their prior
    if fwd.QUENCH: r.append([(x[IDX['qlogC']] - 5.5) / 2.0, (x[IDX['qW']] - 0.5) / 0.2, max(0.0, 0.05 - x[IDX['qW']]) / 0.01])      # weak prior on the quenching parameters
    E, s, logN = ref.build(x[IDX["dE"]], x[IDX["eta"]], x[IDX["dN"]]); ls = np.log10(s); r.append(np.maximum(0, ls - 14.0) / 0.3); r.append(np.maximum(0, 10.0 - ls) / 0.3)
    return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in r])
LAM = dict(tl=1.0, maad=1.0, lab=1.0, itl=1.0, prof=1.0)
LAMFIX = dict((kv.split('=')[0], float(kv.split('=')[1])) for kv in os.environ['LAMFIX'].split(',')) if os.environ.get('LAMFIX') else None      # common, fixed noise factors (same likelihood for every model)
def stack(R, x, lam, robust_w=None):
    parts = []
    for k in ("tl", "maad", "lab", "itl", "prof"):
        r = R[k].ravel() * np.sqrt(KAPPA[k] / lam[k])
        if k == "tl" and robust_w is not None: r = r * np.sqrt(robust_w.ravel())
        parts.append(r)
    parts.append(prior_resid(x)); return np.concatenate(parts)
def evaluate(x, lam, robust_w=None):
    R = fwd.resid_all(ref, x, nj); return stack(R, x, lam, robust_w), R
def update_lambda(R, lam):
    if LAMFIX: return dict(LAMFIX)
    new = {}
    for k in ("tl", "maad", "lab", "itl", "prof"):
        target = max(1.0, float(np.mean(R[k] ** 2))); new[k] = float(np.clip(target, lam[k] / 1.3, lam[k] * 1.3)) if lam[k] > 1.0 or target > 1.0 else 1.0
    return new
def student_w(Rtl, lam_tl): return (NU + 1.0) / (NU + Rtl ** 2 / lam_tl)
H = np.concatenate([np.full(5, 0.01), np.full(5, 0.05), np.full(5, 0.1), [0.02, 0.05], np.full(3, 0.05), [0.02, 0.05], np.full(6, 0.5)])
if NXB >= 29: H = np.append(H, 0.05)
if NXB == 30: H = np.append(H, 0.05)
if fwd.QUENCH: H = np.append(H, [0.1, 0.01])
d0 = json.load(open(f_start)); x = np.array(d0["x"]) if "x" in d0 else None
if x is None:
    p = np.array(d0["best"]); x = np.zeros(NX); x[IDX["logD0"]] = p[12]; x[IDX["T"]] = 45.0
    if fwd.MODEL == "imts": x[IDX["logM0"]] = p[13]; x[IDX["logDnat"]] = p[14:17]; x[IDX["a"]], x[IDX["b"]] = d0["law"]
    else:
        x[IDX["logM0"]] = 9.0; x[IDX["logDnat"]] = p[13:16]; x[IDX["a"]], x[IDX["b"]] = d0.get("law", (-5.5, -0.1))          # independent-trap parameter files: [.., logD0, logDnat(3), (b)]
        if fwd.MODEL in ("gok", "gok2"): x[IDX["bexp"]] = p[16]
        if fwd.MODEL == "gok2": x[IDX["aexp"]] = 1.0
nx0 = len(x)
if len(x) < NX: x = np.concatenate([x, np.ones(NX - len(x))])
if fwd.QUENCH and nx0 < NX: x[IDX['qlogC']] = float(os.environ.get('QC0', 5.5)); x[IDX['qW']] = float(os.environ.get('QW0', 0.5))                                              # warm start from a model with fewer parameters (new ones start at 1)
if os.environ.get('BSTART') and NXB >= 29: x[IDX['bexp']] = float(os.environ['BSTART'])                            # start value of the GOK order
if os.environ.get('DNAT0'): x[IDX['logDnat']] = [float(v) for v in os.environ['DNAT0'].split(',')]                       # start values of the natural-dose nodes (log10 Gy)
if os.environ.get('DE0'): x[IDX['dE']] = float(os.environ['DE0'])                                                 # start value of the energy shifts (eV, all control points)
if os.environ.get('TSTART'): x[IDX['T']] = float(os.environ['TSTART'])          # start temperature of the history (override)
t0 = time.time(); lam = dict(LAM); r, R = evaluate(x, lam); lam = dict(LAMFIX) if LAMFIX else {k: max(1.0, float(np.mean(R[k] ** 2))) for k in LAM}; wtl = student_w(R["tl"], lam["tl"]); r, R = evaluate(x, lam, wtl)
print("lambda_k (estimated):", {k: round(v, 2) for k, v in lam.items()}, f"| cost {0.5 * (r @ r):.1f} | {time.time() - t0:.0f}s", flush=True)
mu = 1e-2; cost = 0.5 * (r @ r)
for it in range(MAXIT):
    t1 = time.time(); J = np.empty((r.size, NX)); 
    for j in range(NX):
        xp = x.copy(); xp[j] += H[j]; rp, _ = evaluate(xp, lam, wtl); J[:, j] = (rp - r) / H[j]
    g = J.T @ r; A = J.T @ J; accepted = False
    for _ in range(6):
        step = np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-9), -g); xn = x + step; rn, Rn = evaluate(xn, lam, wtl); cn = 0.5 * (rn @ rn)
        if cn < cost: x, r, R, cost = xn, rn, Rn, cn; mu = max(mu / 3.0, 1e-6); accepted = True; break
        mu *= 4.0
    lam = update_lambda(R, lam); wtl = student_w(R["tl"], lam["tl"]); r, R = evaluate(x, lam, wtl); cost = 0.5 * (r @ r)
    ch = {k: round(float(np.mean(R[k] ** 2)), 2) if R[k].size else None for k in LAM}
    print(f"iter {it}: cost {cost:.1f} accepted {accepted} mu {mu:.1e} |step| {np.linalg.norm(step):.3f} | mean r^2 {ch} | T {np.round(x[IDX['T']],1).tolist()} | dE {np.round(x[IDX['dE']],3).tolist()} | {time.time()-t1:.0f}s", flush=True)
    json.dump(dict(x=x.tolist(), lam=lam, cost=cost, use_datum=USE_DATUM), open(f"lm_{tag}.json", "w"))
    if accepted and np.linalg.norm(step) < 1e-3: break
# Laplace approximation at the solution
J = np.empty((r.size, NX))
for j in range(NX):
    xp = x.copy(); xp[j] += H[j]; rp, _ = evaluate(xp, lam, wtl); J[:, j] = (rp - r) / H[j]
Cv = np.linalg.pinv(J.T @ J); sd = np.sqrt(np.diag(Cv)); w, V = np.linalg.eigh(J.T @ J)
json.dump(dict(x=x.tolist(), sd=sd.tolist(), lam=lam, cost=cost, use_datum=USE_DATUM, eig=w.tolist(), cov=Cv.tolist()), open(f"lm_{tag}.json", "w"))
print("posterior sd of T (oldest..present):", np.round(sd[IDX["T"]], 2).tolist(), "| present T %.1f +- %.1f" % (x[IDX["T"]][-1], sd[IDX["T"]][-1]), "| dE", np.round(x[IDX["dE"]], 3).tolist(), "+-", np.round(sd[IDX["dE"]], 3).tolist(), flush=True); print("done", flush=True)
