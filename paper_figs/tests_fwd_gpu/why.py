import numpy as np
Z=np.load("ngrip_grid_T5.npz"); T0s,dTs,P,sd=Z["T0s"],Z["dTs"],Z["prof"],Z["sd"]; TR=np.load("truths_T5.npz")
tr={3.:TR["d3"],10.:Z["truth"],15.:TR["d15"]}
def noise(r): return np.interp(np.arange(15),[0,7,14],r.normal(size=3))
def run(d,sdt0,kap=1.0,N=400,flat=False):
    r=np.random.default_rng(1); M=[];Md=[];S=[]
    for _ in range(N):
        data=np.clip(tr[d]+noise(r)*sd,0,1); chi=(((P-data)/sd)**2).sum(-1)
        lp=-0.5*kap*chi-0.5*((T0s[:,None]-5.0)/sdt0)**2-(0 if flat else 0.5*(dTs[None,:]/15)**2); w=np.exp(lp-lp.max()); w/=w.sum(); pd=w.sum(0); cd=np.cumsum(pd)
        m=(pd*dTs).sum(); M.append(m); Md.append(np.interp(0.5,cd,dTs)); S.append(np.sqrt((pd*(dTs-m)**2).sum()))
    return np.mean(M),np.mean(Md),np.std(Md),np.mean(S)
print("true 10 C, 15 windows. T0 prior sd -> mean, median of posterior, scatter of median over noise draws, posterior sd")
for sdt0 in (0.3,0.5,1,2,5): print(" sdT0 %.1f"%sdt0, [round(float(v),1) for v in run(10.,sdt0)])
print("flat dT prior, sdT0 0.5:",[round(float(v),1) for v in run(10.,0.5,flat=True)])
# local sensitivities at T0~5, dT=10 (units of sigma per degree, whole profile, KAP=1)
i=np.argmin(abs(T0s-5.0)); j=np.argmin(abs(dTs-10)); 
dPdT0=(P[i+1,j]-P[i-1,j])/(T0s[i+1]-T0s[i-1]); dPdd=(P[i,j+1]-P[i,j-1])/(dTs[j+1]-dTs[j-1])
# whitened with 3 effective independent points: use all 15 windows with KAP=1
a=dPdT0/sd; b=dPdd/sd; print("sensitivity per degC (sigma): |dP/dT0| %.2f  |dP/ddT| %.2f ; cosine between them %.3f"%(np.linalg.norm(a),np.linalg.norm(b),abs(a@b)/np.linalg.norm(a)/np.linalg.norm(b)))
F=np.array([[a@a,a@b],[a@b,b@b]]); print("Fisher sd(dT) with T0 known %.1f ; with T0 free %.1f"%(1/np.sqrt(F[1,1]),np.sqrt(np.linalg.inv(F)[1,1])))
