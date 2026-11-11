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
temps2=Tp+np.array([3,-4,5,-6,-8,0.0])
for name,fn in (("fill",lambda: eng.fill(edges,temps2)),("kars",lambda: eng.kars(edges,temps2))):
    try: t=time.time(); r=fn(); print("B:",name,"ok %.1fs"%(time.time()-t))
    except Exception as e: print("B:",name,"FAILED",str(e)[:80])
n=eng.fill(edges,temps) 
try: t=time.time(); eng.readout(n); print("B: readout (const) ok %.1fs"%(time.time()-t))
except Exception as e: print("B: readout FAILED",str(e)[:80])
