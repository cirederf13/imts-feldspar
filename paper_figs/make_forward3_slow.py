import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style; import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; style.apply()
import os; HERE=os.path.dirname(os.path.abspath(__file__)); S=HERE+'/inputs/'
rows=[('imts','IMTS'),('gok','GOK'),('a0','$A=0$')]
colmap=lambda l: '#1b6ca8' if l=='best fit' else ('#c0392b' if l in ('m0 x0.1','b=1') else '#2a8f5a')
names={'m0 x0.1':r'$m_0\times0.1$','m0 x10':r'$m_0\times10$','b=1':r'$b=1$','b=2':r'$b=2$'}
fig,ax=plt.subplots(3,3,figsize=(13,10.5)); xs=np.arange(205,346,10); k=0
for r,(key,nm) in enumerate(rows):
    o=pickle.load(open(S+f'f3_{key}.pkl','rb')); labs=o['labels']; p1=np.max(o['glow']['best fit'])
    for l in labs:
        lab=names.get(l, ('best fit'+(f' ($b={o["b"]:.2f}$)' if key=='gok' else '')))
        ax[r,0].semilogx(o['Dg'],o['growth'][l],color=colmap(l),label=lab)
        ax[r,1].plot(o['Tgrid'],o['glow'][l]/p1,color=colmap(l))
        ax[r,2].plot(xs,o['prof'][l],color=colmap(l),lw=1.8)
    ax[r,2].errorbar(xs,o['data'],o['sd'],fmt='ko',ms=3.5,capsize=2)
    ax[r,0].set_ylabel(f'{nm}\nTL area, norm. at 1 kGy'); ax[r,0].legend(fontsize=8)
    ax[r,1].set(xlim=(60,450),ylabel='TL / peak of best fit'); ax[r,2].set(ylabel='n/N'); ax[r,2].set_title(f'{o["T0"]:.1f} °C',fontsize=9)
    for c in range(3):
        ax[r,c].grid(alpha=.3); ax[r,c].text(0.02,0.95,'(%s)'%'abcdefghi'[k],transform=ax[r,c].transAxes,va='top',fontsize=11,fontweight='bold'); k+=1
for c,t in enumerate(['Growth with dose','Glow curve after 300 Gy','Natural profile']): ax[0,c].set_title(t if c<2 else 'Natural profile, '+f'{pickle.load(open(S+"f3_imts.pkl","rb"))["T0"]:.1f} °C',fontsize=10)
for c,x in enumerate(['β dose (Gy)','Temperature (°C)','TL window (°C)']): ax[2,c].set_xlabel(x)
fig.tight_layout(); fig.savefig(HERE+'/out/forward3_slow.png',dpi=110)
Dg=None
for key,nm in rows:
    o=pickle.load(open(S+f'f3_{key}.pkl','rb')); Dg=o['Dg']
    print(key,'T0 %.1f'%o['T0'],'chi2',{a:round(b,1) for a,b in o['chi2'].items()})
    for l in o['labels']:
        g=o['growth'][l]; gl=o['glow'][l]; b=o['glow']['best fit']; T=o['Tgrid']
        print('  ',l,'growth16k/1k %.2f'%np.interp(16000,Dg,g),'peakT %.0f'%T[np.argmax(gl)],'rel@300C %.2f'%(np.interp(300,T,gl)/np.interp(300,T,b)))
