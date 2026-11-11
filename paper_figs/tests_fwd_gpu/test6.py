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
ref_p=NS["hist_model"](temps,sat)
t=time.time(); p=np.asarray(eng.profile(edges,temps)); print("A: jax %.1fs (incl compile)"%(time.time()-t))
t=time.time(); p=np.asarray(eng.profile(edges,temps)); print("A: jax 2nd %.2fs"%(time.time()-t))
print("A: max |d n/N| = %.2e   (sd min %.3f);  max|d|/sd = %.4f"%(np.abs(p-ref_p).max(), HA.NN_SD.min(), (np.abs(p-ref_p)/HA.NN_SD).max()))
import diffrax, jax.numpy as jnp
def tryfill(temps_, rtol=1e-8, ms=4000000):
    e=jnp.asarray(edges); T=jnp.asarray(temps_)+273.15; Tf=lambda t: T[jnp.clip(jnp.searchsorted(e,t,side="right")-1,0,T.size-1)]
    A,B,M0,N,E,s_,G=eng.A*3.15576e7,eng.B*3.15576e7,eng.M0,eng.N,eng.E,eng.s,eng.G
    def rhs(t,n,_):
        lam=s_*jnp.exp(-E/(8.617333262e-5*Tf(t)))*3.15576e7; nc=(jnp.sum(lam*n)+G)/(A*jnp.sum(N-n)+B*(M0+jnp.sum(n))); return -lam*n+A*nc*(N-n)
    ctrl=diffrax.PIDController(rtol=rtol,atol=eng.atol_hold,pcoeff=0.3,icoeff=0.4,jump_ts=e[1:-1])
    sol=diffrax.diffeqsolve(diffrax.ODETerm(rhs),diffrax.Kvaerno5(),0.0,float(e[-1]),1.0,jnp.zeros_like(N),stepsize_controller=ctrl,max_steps=ms,throw=False,saveat=diffrax.SaveAt(t1=True))
    return sol
for lab,dT in (("const",[0,0,0,0,0,0]),("+3 first",[3,0,0,0,0,0]),("-4 second",[0,-4,0,0,0,0]),("last -8",[0,0,0,0,-8,0]),("all B",[3,-4,5,-6,-8,0])):
    sol=tryfill(Tp+np.array(dT,float)); print("B: %-10s result %s steps %d"%(lab,sol.result,int(sol.stats["num_steps"])),flush=True)
