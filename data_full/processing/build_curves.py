"""Group parsed BIN records into per-(rate, preheat, position) curve sets.
Per set/position the sequence is: [TL450 conditioning x2 (set 1/first only)],
beta(1916s) -> TL to PH -> TL450 (residual) -> TL450 (bg) -> beta(958s test) ->
TL to PH -> TL450 (test residual) -> TL450 (test bg)."""
import sys, collections, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_bin import read_bin
U = os.environ.get("RAW_DIR", "raw") + "/"       # folder with the raw .binx files
FILES = [U+"R3_ANA-05_NaF_TL-VHR_cups.binx", U+"R3_ANA-05_NaF_TL-VHR_cups_5deg.binx"]
POS = {17: 'cup17', 19: 'cup19', 21: 'cup21'}

def build():
    out = {}   # (rate, ph, pos) -> dict(reg, reg_bg, tst, tst_bg) arrays
    for fi, f in enumerate(FILES):
        groups = collections.defaultdict(list)
        for r in read_bin(f):
            groups[(r['set'], r['pos'])].append(r)
        for (s, pos), recs in groups.items():
            recs = [r for r in sorted(recs, key=lambda r: r['run'])
                    if r['run'] > 2]   # drop conditioning TL450s
            if fi == 0 and s == 1:      # main-file set 1: no-preheat reference (TL450 after 300 Gy, no PH/test-dose steps)
                out[('ref450', 0, pos)] = dict(recs=[r['data'] for r in recs]); continue
            if len(recs) != 6:
                print("WARN set", s, "pos", pos, "has", len(recs), "records", [(r['run'], r['tmax'], r['rate']) for r in recs]); continue
            ph_reg, reg, bg, ph_t, tst, tbg = recs
            assert ph_reg['tmax'] == ph_t['tmax'] and reg['npts'] == 450 and abs(ph_reg['rate']-reg['rate']) < 1e-6
            assert ph_reg['irr'] == 1916 and ph_t['irr'] == 958, (s, pos, ph_reg['irr'], ph_t['irr'])
            key = (round(float(reg['rate']), 3), int(ph_reg['tmax']), pos)
            if key in out: print("DUP", key, "file", fi, "set", s)
            out[key] = dict(set=s, ph_reg=ph_reg['data'], reg=reg['data'], bg=bg['data'], ph_t=ph_t['data'], tst=tst['data'], tbg=tbg['data'])
    return out
if __name__ == '__main__':
    o = build()
    o_ref = {k: o.pop(k) for k in [k for k in o if k[0]=='ref450']}
    rates = sorted(set(k[0] for k in o)); phs = sorted(set(k[1] for k in o))
    print("rates", rates); print("preheats", phs, len(phs)); print("n keys", len(o), "expected", len(rates)*len(phs)*3)
    print(f"{'rate':>5} {'ph':>4} {'pos':>3} {'set':>3} {'reg_peak':>9} {'bg_mean':>8} {'tst_peak':>9} {'T_pk(C)':>7}")
    for k in sorted(o):
        d = o[k]; 
        if k[2] == 17 or True:
            if k[1] in (60, 240, 420) : print(f"{k[0]:5g} {k[1]:4d} {k[2]:3d} {d['set']:3d} {d['reg'].max():9.0f} {d['bg'].mean():8.1f} {d['tst'].max():9.0f} {int(d['reg'].argmax()):7d}")

def write_txt(outdir='.', prefix='ana05cups'):
    """Write raw post-preheat TL450 curves in the project's tl_data_*.txt format
    (T[K]=channel index(0..449)+273.15, raw counts, same as old xlsx-derived files)."""
    o = build(); n = 0
    for (rate, ph, pos), d in o.items():
        if rate == 'ref450': continue
        cup = {17: 'cup17', 19: 'cup19', 21: 'cup21'}[pos]
        fn = f"{outdir}/tl_data_{prefix}_{cup}_ph{ph}_b{rate:g}.txt"
        with open(fn, 'w') as f:
            f.write("Temperature(K) TL_Intensity\n")
            for i, v in enumerate(d['reg']):
                f.write(f"{i + 273.15:.15e} {v:.15e}\n")
        n += 1
    return n

def write_txcorrected(outdir='.', prefix='ana05cupsTx'):
    """Test-dose (sensitivity) corrected curves.  Tx = sum_{ch>=70}(test-dose TL450 - its bg).
    Data = reg * f,  f = Tref/Tx,  Tref = geometric mean of Tx over the 6 rates for that (cup,preheat),
    so the data stay in count-like units.  Factors saved in tx_factors.json for exact variance propagation:
    var(data) = f^2*a + f*b*data   (a,b = noise_model_ana05.json)."""
    import json
    o = build(); o = {k: v for k, v in o.items() if k[0] != 'ref450'}
    Tx = {k: float((d['tst'] - d['tbg'])[70:].sum()) for k, d in o.items()}
    fac = {}
    for (rate, ph, pos), d in o.items():
        grp = [Tx[(r, ph, pos)] for r in sorted(set(k[0] for k in o))]
        Tref = float(np.exp(np.mean(np.log(grp))))
        f = Tref / Tx[(rate, ph, pos)]
        cup = {17: 'cup17', 19: 'cup19', 21: 'cup21'}[pos]
        fac[f"{cup}_ph{ph}_b{rate:g}"] = dict(Tx=Tx[(rate, ph, pos)], Tref=Tref, fac=f)
        with open(f"{outdir}/tl_data_{prefix}_{cup}_ph{ph}_b{rate:g}.txt", 'w') as fh:
            fh.write("Temperature(K) TL_Intensity\n")
            for i, v in enumerate(d['reg'] * f):
                fh.write(f"{i + 273.15:.15e} {v:.15e}\n")
    json.dump(fac, open(f"{outdir}/newdata_vhr_cups/tx_factors.json", 'w'), indent=1)
    return len(fac)
