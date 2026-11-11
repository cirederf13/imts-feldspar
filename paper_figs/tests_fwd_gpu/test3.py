import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IM=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "start_joint1_final.json"); os.environ["IMTS_PARAMS"]=IM
import fwd, pgrid, jax, jax.numpy as jnp, diffrax
import hold_jax as H
fwd.init_main(); ref=pgrid.Ref(IM); IDX=fwd.IDX; SI=fwd.SI; GI=fwd.GI
x=np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lm_lmBs.json")))["x"]); m=fwd.model_arrays(ref,x); fwd.apply_SI(m,x[IDX["a"]],x[IDX["b"]])
Tp=x[IDX["T"]][-1]
# 1 Ma at constant Tp-10, then 120 kyr in 6000 stages of 20 yr with a smooth synthetic D-O-like record (+-4 C, 1.5 kyr period) and a 10 C rise 19-11.7 ka
tt=np.arange(6000)*20.0; age=120e3-tt-10.0
rec=-10*np.clip((age-11.7e3)/(19e3-11.7e3),0,1)*(age<=1e9)+4*np.sin(2*np.pi*age/1500.0)*(age>11.7e3)
temps=Tp+rec
n=SI.hold(np.zeros(SI.NT),Tp-10+273.15,1.0,SI.DOT); n0=n.copy()
hold=H.make_hold(SI.E,SI.s,SI.N,SI.D0,GI.M0,A=GI.A_val,B=GI.B,rtol=1e-8)
edges=np.arange(6000)*20.0
Tf=H.piecewise_T(edges,temps)
# jump times for the stepper
import functools
def run(n_start, temps_):
    Tf=H.piecewise_T(edges,temps_)
    Ee,ss,NN=map(jnp.asarray,(SI.E,SI.s,SI.N)); sumN=NN.sum(); G=SI.DOT/1000.0/SI.D0*sumN; A=GI.A_val*H.YR; B=GI.B*H.YR; M0=GI.M0
    def rhs(t,n,args):
        lam=ss*jnp.exp(-Ee/(H.kB*Tf(t)))*H.YR; nc=(jnp.sum(lam*n)+G)/(A*jnp.sum(NN-n)+B*(M0+jnp.sum(n))); return -lam*n+A*nc*(NN-n)
    ctrl=diffrax.PIDController(rtol=1e-8,atol=1e-6*float(SI.N.min()),pcoeff=0.3,icoeff=0.4,jump_ts=jnp.asarray(edges[1:]))
    sol=diffrax.diffeqsolve(diffrax.ODETerm(rhs),diffrax.Kvaerno5(),0.0,120e3,1.0,jnp.asarray(n_start),stepsize_controller=ctrl,max_steps=2000000,saveat=diffrax.SaveAt(t1=True))
    return jnp.clip(sol.ys[-1],0,NN), sol.stats["num_steps"]
# batch over 32 scale factors with vmap

import sys
for B in (128,512):
    scales=np.linspace(0.5,1.5,B); f=jax.jit(jax.vmap(lambda sc: run(n0, Tp+sc*rec)[0]))
    t=time.time(); out=f(jnp.asarray(scales)); out.block_until_ready(); t1=time.time()-t
    t=time.time(); out=f(jnp.asarray(scales)); out.block_until_ready(); print("vmap %d second call: %.1fs = %.3f s/history (first %.0fs)"%(B,time.time()-t,(time.time()-t)/B,t1),flush=True)
