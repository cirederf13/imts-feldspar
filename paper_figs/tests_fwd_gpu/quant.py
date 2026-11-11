import numpy as np
Z=np.load("ngrip_grid_T5.npz"); TR=np.load("truths_T5.npz")
T0s,dTs,P,sd=Z["T0s"],Z["dTs"],Z["prof"],Z["sd"]; KAP=0.2
tr={0.:TR["d0"],3.:TR["d3"],5.:TR["d5"],10.:Z["truth"],15.:TR["d15"]}
print("window n/N truth profiles (T0=5):"); 
for d,p in tr.items(): print(d, np.round(p[::2],3))
print("misfit distance from dT=0 profile  sqrt(KAP*chi2), and per-degree increment")
ds=sorted(tr); D=[np.sqrt(KAP*(((tr[d]-tr[0.])/sd)**2).sum()) for d in ds]
for i,d in enumerate(ds): print("dT %4.1f  dist %.2f  incr/deg %.3f"%(d,D[i],(D[i]-D[i-1])/(d-ds[i-1]) if i else 0))
# grid: sensitivity along dT at T0=5 (grid row nearest)
i5=np.argmin(abs(T0s-5)); print("T0 row",T0s[i5])
dd=[np.sqrt(KAP*(((P[i5,j]-P[i5,np.argmin(abs(dTs-0))])/sd)**2).sum()) for j in range(dTs.size)]
print("grid dist from dT=0:",dict(zip(dTs,np.round(dd,2))))
# prior effect: noise-averaged posterior mean, T0 known +-2, three priors
def noise(r): return np.interp(np.arange(15),[0,7,14],r.normal(size=3))
def run(prior,kap=KAP,N=400):
    out={}
    for d in ds:
        r=np.random.default_rng(100+int(d)); m=[]
        for _ in range(N):
            data=np.clip(tr[d]+noise(r)*sd,0,1); chi=(((P-data)/sd)**2).sum(-1)
            lp=-0.5*kap*chi-0.5*((T0s[:,None]-5.0)/2.0)**2
            if prior=="gauss": lp=lp-0.5*(dTs[None,:]/15)**2
            elif prior=="pos": lp=lp+np.where(dTs[None,:]<0,-1e9,0)
            w=np.exp(lp-lp.max()); w/=w.sum(); m.append((w.sum(0)*dTs).sum())
        out[d]=np.mean(m)
    return out
for pr in ("flat","gauss","pos"): print(pr,{d:round(v,1) for d,v in run(pr).items()})
print("15 windows flat",{d:round(v,1) for d,v in run("flat",1.0).items()})
