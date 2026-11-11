import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IM=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "start_joint1_final.json"); os.environ["IMTS_PARAMS"]=IM
import fwd, pgrid, jax.numpy as jnp
from history_jax import Engine
fwd.init_main(); ref=pgrid.Ref(IM); IDX=fwd.IDX; SI,GI,HA,NS=fwd.SI,fwd.GI,fwd.HA,fwd.NS
x=np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lm_lmBs.json")))["x"]); m=fwd.model_arrays(ref,x); fwd.apply_SI(m,x[IDX["a"]],x[IDX["b"]])
sat=NS["lab_sat"](); eng=Engine(SI,GI,HA,sat)
Tp=float(x[IDX["T"]][-1])
# case A: the six-stage inversion history (as in the fits), oldest -> youngest, constant Tp
seg=[(3000.,10000.),(1000.,3000.),(100.,1000.),(30.,100.),(10.,30.),(0.,10.)]
edges=np.concatenate([[0.0],np.cumsum([(b-a)*1e3 for a,b in seg])]); temps=np.array([Tp,Tp,Tp,Tp,Tp,Tp])
import diffrax, jax, jax.numpy as jnp
YR=3.15576e7; kB=8.617333262e-5
A,B,M0,N,E,s_,G=eng.A*YR,eng.B*YR,eng.M0,eng.N,eng.E,eng.s,eng.G
def stage(n,Tk,dtk,rt=1e-8,dt0=1.0,solver=None):
    lam=s_*jnp.exp(-E/(kB*Tk))*YR
    def rhs(t,y,_):
        nc=(jnp.sum(lam*y)+G)/(A*jnp.sum(N-y)+B*(M0+jnp.sum(y))); return -lam*y+A*nc*(N-y)
    ctrl=diffrax.PIDController(rtol=rt,atol=eng.atol_hold,pcoeff=0.3,icoeff=0.4)
    sol=diffrax.diffeqsolve(diffrax.ODETerm(rhs),solver or diffrax.Kvaerno5(),0.0,dtk,dt0,n,stepsize_controller=ctrl,max_steps=20000,throw=False,saveat=diffrax.SaveAt(t1=True))
    return jnp.clip(sol.ys[-1],0,N), int(sol.stats["num_steps"]), sol.result==diffrax.RESULTS.successful
print("A: atol",eng.atol_hold,"Nmin",float(N.min()),"Nmax",float(N.max()),flush=True)
n=jnp.zeros_like(N)
for k,(dT,dtk) in enumerate([(0,7e6),(0,2e6),(0,9e5),(3,7e4),(-4,2e4),(0,1e4)]):
    n,ns,ok=stage(n,Tp+273.15+dT,dtk); print("A: stage",k,"dT",dT,"dt %.0e steps %d %s"%(dtk,ns,"ok" if ok else "FAIL"),flush=True)
