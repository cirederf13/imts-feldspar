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
def stage(x,Tk,dtk,rt=1e-8,atol=1e-10,dt0=1.0,dtmax=None):
    lam=s_*jnp.exp(-E/(kB*Tk))*YR
    def rhs(t,y,_):
        n=y*N; nc=(jnp.sum(lam*n)+G)/(A*jnp.sum(N-n)+B*(M0+jnp.sum(n))); return -lam*y+A*nc*(1-y)
    ctrl=diffrax.PIDController(rtol=rt,atol=atol,pcoeff=0.3,icoeff=0.4,dtmax=dtmax)
    sol=diffrax.diffeqsolve(diffrax.ODETerm(rhs),diffrax.Kvaerno5(),0.0,dtk,dt0,x,stepsize_controller=ctrl,max_steps=20000,throw=False,saveat=diffrax.SaveAt(t1=True))
    return jnp.clip(sol.ys[-1],0,1), int(sol.stats["num_steps"]), sol.result==diffrax.RESULTS.successful
for dtmax in (1e4,1e5,1e6):
  x=jnp.zeros_like(N)
  for k,(dT,dtk) in enumerate([(0,7e6),(0,2e6),(0,9e5),(3,7e4),(-4,2e4),(0,1e4)]):
    x,ns,ok=stage(x,Tp+273.15+dT,dtk,dtmax=dtmax); print("A: dtmax %.0e stage %d dT %d steps %d %s"%(dtmax,k,dT,ns,"ok" if ok else "FAIL"),flush=True)
