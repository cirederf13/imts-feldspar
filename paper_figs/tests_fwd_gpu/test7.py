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
lam=s_*jnp.exp(-E/(kB*(Tp+273.15)))*YR
def rhs(t,y,_):
    nc=(jnp.sum(lam*y)+G)/(A*jnp.sum(N-y)+B*(M0+jnp.sum(y))); return -lam*y+A*nc*(N-y)
for dtk in (1e3,1e5,7e6):
  for rt in (1e-8,1e-6):
    ctrl=diffrax.PIDController(rtol=rt,atol=eng.atol_hold,pcoeff=0.3,icoeff=0.4)
    sol=diffrax.diffeqsolve(diffrax.ODETerm(rhs),diffrax.Kvaerno5(),0.0,dtk,1.0,jnp.zeros_like(N),stepsize_controller=ctrl,max_steps=100000,throw=False,saveat=diffrax.SaveAt(t1=True))
    print("A: dt %.0e rtol %.0e steps %d %s"%(dtk,rt,int(sol.stats["num_steps"]),"ok" if sol.result==diffrax.RESULTS.successful else "FAIL"),flush=True)
