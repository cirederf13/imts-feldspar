"""KTB20-16 (Na-feldspar) data terms for the stacked inversion (replaces the ANA-05 data terms in fwd): Tm-Tstop shapes at 1 C/s, ITL Lx/Tx per 10 C window,
MAAD growth in five windows with natural doses, natural n/N in ten windows 200-299 C, rho' per window fixed from the fading analysis (no fading data term).
Data are read from a local pickle (made by ktb_extract.py) and are not part of this file.  usage: KTB=<pickle> python lm_inv.py ..."""
import pickle, numpy as np, fwd
M = fwd.M; G = fwd.G; HA = fwd.HA
WIN = [(200 + 10 * k, 210 + 10 * k) for k in range(10)]; MW = [1, 3, 5, 7, 9]               # MAAD windows used (indices into WIN)
import os
DDOT_NAT = 2.26; DR_LAB = 0.1500; BETA_I = 3
if os.environ.get('MIZ'): DDOT_NAT = float(os.environ.get('MIZ_DDOT', 8.44)); DR_LAB = 0.1276                                 # MIZ-1-1 (K-feldspar): env dose rate, MAAD reader dose rate                                                 # natural dose rate (Gy/ka), MAAD reader dose rate, 1 C/s
def apply(pkl):
    D = pickle.load(open(pkl, 'rb')); Tg = np.asarray(M.T_surr, float) - 273.15; mask = (Tg >= 70) & (Tg <= 440); ch = np.arange(1, 451)
    # ---- Tm-Tstop: peak-normalised shapes, three cups x 15 Tstop
    TS = list(range(160, 301, 10)); POS = (19, 21, 23) if os.environ.get('MIZ') else (43, 45, 47); tl_y = np.zeros((3, 15, mask.sum())); tl_sd = np.zeros_like(tl_y)
    for i, p in enumerate(POS):
        for j, t in enumerate(TS):
            c = np.interp(Tg, ch, D['tt'][(p, t)]); sm = np.convolve(c, np.ones(5) / 5, mode='same'); y = c / sm.max(); ys = sm / sm.max()
            tl_y[i, j] = y[mask]; tl_sd[i, j] = np.sqrt(np.abs(ys[mask]) / 2000.0 + (0.03 * ys[mask]) ** 2 + (3.0 * np.gradient(ys, Tg)[mask]) ** 2 + 1e-8)
    def tl_resid(m, nj):
        res = fwd.Parallel(n_jobs=nj)(fwd.delayed(fwd._stage)(m['E'], m['s'], m['N'], m['D0'], m['M0'], float(t), D['tt_dose'], [BETA_I], m['b'], m['a']) for t in TS)
        mod = np.stack([r[0][0][mask] for r in res]); return (mod[None] - tl_y) / tl_sd
    # ---- ITL: mean over cups of (Lx/Tx)(T, hold) / (Lx/Tx)(150, 0), per window
    THT = [150 + 20 * k for k in range(11)]; HT = [0.0, 10, 30, 100, 300, 1000]; ref = np.mean([D['itl'][(p, 150, 0.0)] for p in (1, 3, 5)], 0)
    jobs = [(T, th) for T in THT for th in HT if not (T == 150 and th == 0.0)]; itl_y = {}; itl_sel = {}
    for T, th in jobs:
        v = np.array([D['itl'][(p, T, th)] / D['itl'][(p, 150, 0.0)] for p in (1, 3, 5)]); itl_y[(T, th)] = v.mean(0); itl_sel[(T, th)] = v.mean(0) > 0.02
    def itl_resid(m, nj):
        out = fwd.Parallel(n_jobs=nj)(fwd.delayed(fwd._itl_cond)(m['E'], m['s'], m['N'], m['D0'], m['M0'], T, th, m['b'], m['a'], D['itl_dose']) for T, th in [(150, 0.0)] + jobs)
        r0 = out[0]; r = []
        for (T, th), mm in zip(jobs, out[1:]):
            y = itl_y[(T, th)]; sg = np.sqrt((0.05 * y) ** 2 + 0.01 ** 2); r.append(((mm / r0 - y) / sg)[itl_sel[(T, th)]])
        return np.concatenate(r)
    # ---- MAAD growth: windows MW, three groups, doses natural + added; natural dose per window interpolated (log) between three nodes
    Dg = D['maad_D']; PHC = 200.0 if os.environ.get('MIZ') else 180.0; wts = [((Tg >= WIN[j][0]) & (Tg < WIN[j][1])).astype(float) for j in MW]; node_j = np.array([MW[0], MW[2], MW[4]], float)
    def maad_resid(m, nj):
        lDn = np.interp(np.array(MW, float), node_j, np.log10(m['Dnat'])); keys = [(jj, float(Dg[k] + 10.0 ** lDn[jj])) for jj in range(len(MW)) for k in range(8)]
        res = fwd.Parallel(n_jobs=nj)(fwd.delayed(fwd._stage)(m['E'], m['s'], m['N'], m['D0'], m['M0'], PHC, d, [BETA_I], m['b'], m['a']) for _, d in keys); r = []
        for jj in range(len(MW)):
            f = np.array([10.0 ** res[jj * 8 + k][0][1] * np.sum(res[jj * 8 + k][0][0] * wts[jj]) for k in range(8)]); f = f / f.max()
            for g in range(3):
                y = D['maad'][g, MW[jj]]; y = y / y.max(); sg = 0.10 * y + 0.03; w = 1.0 / sg ** 2; A = np.sum(w * f * y) / np.sum(w * f * f); r.append((A * f - y) / sg)
        return np.concatenate(r)
    # ---- fading: rho' per window fixed (no data term); natural profile from the history
    nodes = np.array([0.5 * (lo + hi) for lo, hi in WIN]); logrho = np.log10(D['rho']); apply_SI0 = fwd.apply_SI
    def apply_SI(m, a, b):
        HA.LOGRHO_T = np.zeros(4); HA.LOGRHO_V = np.zeros(4); apply_SI0(m, a, b); HA.LOGRHO_T = nodes.copy(); HA.LOGRHO_V = logrho.copy(); fwd.SI.W_K, fwd.SI.LA_K = fwd.SI.rate_classes(fwd.SI.E, fwd.SI.s)
    HA.win = lambda I: np.array([I[(HA.TC >= lo) & (HA.TC < hi)].sum() for lo, hi in WIN]); HA.NN_DATA = np.asarray(D['nN'], float); HA.NN_SD = 0.02 + 0.08 * HA.NN_DATA
    src = open('inv_joint.py').read(); seg = src[src.index('def lab_sat'):src.index('def prior_T')].replace('0.1566', str(DR_LAB)).replace('19950.0', str(round(3124.0 / DR_LAB, 1)))
    exec(compile(seg, 'seg_ktb', 'exec'), fwd.NS)
    def prof_resid(T):
        sat = fwd.NS['lab_sat'](); r = (fwd.NS['hist_model'](np.asarray(T, float), sat, DDOT_NAT) - HA.NN_DATA) / HA.NN_SD / float(os.environ.get('PROF_SDX', 1))
        return r[::int(os.environ.get('PROF_STEP', 1))]
    fwd.tl_resid, fwd.itl_resid, fwd.maad_resid, fwd.apply_SI, fwd.prof_resid = tl_resid, itl_resid, maad_resid, apply_SI, prof_resid
    fwd.lab_resid = lambda: np.zeros(1)
