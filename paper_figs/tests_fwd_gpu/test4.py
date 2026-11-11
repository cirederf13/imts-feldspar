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
# case C: fine history, last 130 kyr in 100-yr stages (ramp 19->11.7 ka, -10 C before) after 1 Ma
ages=np.r_[np.arange(1_000_000, 130_000, -10_000.0), np.arange(130_000, 0, -100.0)]; dts=np.r_[np.full(87,10_000.0),np.full(1300,100.0)]; tm=ages-dts/2
rec=-10*np.clip((tm-11700.)/(19000.-11700.),0,1); temps3=Tp+rec
ed3=np.concatenate([[0.0],np.cumsum(dts)])
NS["DUR_MYR"]=list(dts/1e6)
t=time.time(); ref3=NS["hist_model"](temps3,sat); print("C: scipy/numpy path %.1fs"%(time.time()-t),flush=True)
t=time.time(); p3=np.asarray(eng.profile(ed3,temps3)); print("C: jax incl compile %.1fs"%(time.time()-t))
t=time.time(); p3=np.asarray(eng.profile(ed3,temps3)); print("C: jax 2nd %.2fs"%(time.time()-t))
print("C: max |d n/N| = %.2e;  max|d|/sd = %.4f"%(np.abs(p3-ref3).max(), (np.abs(p3-ref3)/HA.NN_SD).max()))

p4=np.asarray(eng.profile(ed3,temps3)) 
import history_jax
n_lin=eng.fill(np.concatenate([[0.0],np.cumsum(dts)]),np.r_[temps3,temps3[-1]],mode="linear"); n_st=eng.fill(ed3,temps3); print("C: |n_linear - n_steps|/N max %.2e"%float(np.max(np.abs(np.asarray(n_lin)-np.asarray(n_st))/SI.N)))
