import numpy as np
ZE=np.load("epica_grid_T5.npz"); ZN=np.load("ngrip_grid_T5.npz"); TR=np.load("truths_T5.npz"); DS=[0.,3.,5.,10.,15.]
tN={0.:TR["d0"],3.:TR["d3"],5.:TR["d5"],10.:ZN["truth"],15.:TR["d15"]}; tE=dict(zip(DS,ZE["truths"]))
def noise(r): return np.interp(np.arange(15),[0,7,14],r.normal(size=3))
def q(pd,g,ps): return np.interp(ps,np.cumsum(pd),g)
def run(Z,tr,N=500):
    T0s,dTs,P,sd=Z["T0s"],Z["dTs"],Z["prof"],Z["sd"]; out={}
    for d in DS:
        r=np.random.default_rng(100+int(d)); R=[]
        for _ in range(N):
            data=np.clip(tr[d]+noise(r)*sd,0,1); chi=(((P-data)/sd)**2).sum(-1); lp=-0.5*chi-0.5*((T0s[:,None]-5.0)/2.0)**2-0.5*(dTs[None,:]/15)**2
            w=np.exp(lp-lp.max()); w/=w.sum(); pd=w.sum(0); R.append(q(pd,dTs,[.025,.16,.5,.84,.975]))
        R=np.array(R); out[d]=(R[:,2].mean(),R[:,2].std(),R[:,1].mean(),R[:,3].mean(),np.mean(R[:,0]>0),np.mean((R[:,0]<d)&(R[:,4]>d)))
    return out
for nm,Z,tr in (("NGRIP truth, NGRIP shape",ZN,tN),("EPICA truth, EPICA shape",ZE,tE),("NGRIP truth, EPICA shape",ZE,tN)):
    print(nm); print("true | median (scatter over draws) | 68% interval | excl.0 | cov95")
    for d,v in run(Z,tr).items(): print("%4.0f | %5.1f (%.1f) | %5.1f .. %5.1f | %3.0f%% | %3.0f%%"%(d,v[0],v[1],v[2],v[3],100*v[4],100*v[5]))
