import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IM=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "start_joint1_final.json"); os.environ["IMTS_PARAMS"]=IM
import fwd, pgrid, jax.numpy as jnp
import hold_jax as H
fwd.init_main(); ref=pgrid.Ref(IM); IDX=fwd.IDX; SI=fwd.SI; GI=fwd.GI
x=np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lm_lmBs.json")))["x"]); m=fwd.model_arrays(ref,x); fwd.apply_SI(m,x[IDX["a"]],x[IDX["b"]])
print("NT",SI.NT,"M0 %.3g D0 %.1f B/A %.2g"%(GI.M0,SI.D0,GI.B/GI.A_val))
hold=H.make_hold(SI.E,SI.s,SI.N,SI.D0,GI.M0,A=GI.A_val,B=GI.B)
T0=x[IDX["T"]][-1]+273.15
# test 1: constant T, 1 Ma from empty
t=time.time(); ref1=SI.hold(np.zeros(SI.NT),T0,1.0,SI.DOT); print("scipy const 1 Ma: %.2fs"%(time.time()-t))
Tc=lambda tt: T0+0*tt
t=time.time(); n1,ns=hold(np.zeros(SI.NT),0.0,1e6,Tc); n1=np.asarray(n1); print("jax const 1 Ma: %.2fs steps %d"%(time.time()-t,int(ns)))
t=time.time(); n1,ns=hold(np.zeros(SI.NT),0.0,1e6,Tc); print("jax (2nd call) %.2fs"%(time.time()-t))
print(" max |dn|/N", np.abs(n1-ref1/1).max() if False else float(np.max(np.abs(np.asarray(n1)-ref1)/SI.N)))
# test 2: stepped history with 40 stages (cold LGM-like), scipy sequential vs jax single ODE
edges=np.concatenate([[0.0],np.cumsum(np.r_[np.full(20,30e3),np.full(20,1e3)])]); temps=np.r_[np.full(20,T0-273.15-10),np.full(10,T0-273.15-10),np.full(10,T0-273.15)]
n=np.zeros(SI.NT); t=time.time()
for k in range(40): n=SI.hold(n,temps[k]+273.15,(edges[k+1]-edges[k])/1e6,SI.DOT)
print("scipy 40 stages %.2fs"%(time.time()-t))
Tf=H.piecewise_T(edges[:-1],temps); t=time.time(); n2,ns=hold(np.zeros(SI.NT),0.0,edges[-1],Tf); n2=np.asarray(n2); print("jax 40 stages (one ODE) %.2fs steps %d"%(time.time()-t,int(ns)))
print(" max |dn|/N", float(np.max(np.abs(n2-n)/SI.N)))
