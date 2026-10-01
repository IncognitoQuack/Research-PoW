import sys, numpy as np, torch, random, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/claude/rev'); from data import load_fold
from distortion import apsp, pdist_ball
plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
def embed(D, steps=600, seed=0):
    torch.manual_seed(seed); n=len(D); Dt=torch.tensor(D,dtype=torch.float32); mask=torch.triu(torch.ones(n,n),1)>0
    v=torch.nn.Parameter(0.1*torch.randn(n,2)); s=torch.nn.Parameter(torch.tensor(0.)); opt=torch.optim.Adam([v,s],lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); nv=v.norm(dim=-1,keepdim=True).clamp(min=1e-7); x=torch.tanh(nv).clamp(max=1-1e-5)*v/nv
        d=pdist_ball(x)*torch.exp(s); loss=(((d-Dt)/Dt.clamp(min=1))**2)[mask].mean(); loss.backward(); opt.step()
    with torch.no_grad():
        nv=v.norm(dim=-1,keepdim=True).clamp(min=1e-7); x=torch.tanh(nv)*v/nv
        # Mobius-translate root to origin for readability
        a=x[0:1]; x2=(x*x).sum(-1,keepdim=True); a2=(a*a).sum(); xa=(x*(-a)).sum(-1,keepdim=True)
        num=(1+2*xa+x2)*(-a)+(1-a2)*x; den=1+2*xa+a2*x2; x=num/den
    return x.numpy()
G,_,_=load_fold('twitter16',0); rng=random.Random(3)
names=['Non-rumour','True rumour','False rumour','Unverified']
fig,axes=plt.subplots(1,4,figsize=(7.0,1.95))
for c,ax in enumerate(axes):
    cands=[e for e in G if G[e]['y']==c and 25<=len(G[e]['par'])<=60 and np.bincount([0]+[1]*0).size and True]
    # prefer a cascade with at least depth 3 when available, otherwise any
    def dep(g):
        d=np.zeros(len(g['par']),int)
        for i in range(1,len(g['par'])): d[i]=d[g['par'][i]]+1
        return d
    cands.sort(key=lambda e:(-dep(G[e]).max(), e)); e=cands[rng.randrange(min(8,len(cands)))]
    g=G[e]; d=dep(g); x=embed(apsp(g['par']))
    th=np.linspace(0,2*np.pi,200); ax.plot(np.cos(th),np.sin(th),'k-',lw=0.7)
    for i in range(1,len(g['par'])):
        p=g['par'][i]; ax.plot([x[p,0],x[i,0]],[x[p,1],x[i,1]],'-',color='#bbb',lw=0.5,zorder=1)
    sc=ax.scatter(x[:,0],x[:,1],c=d,cmap='YlOrRd',vmin=0,vmax=max(3,d.max()),s=9,edgecolors='k',linewidths=0.2,zorder=2)
    r=np.linalg.norm(x,axis=1).mean()
    ax.set_title(f'{names[c]}\n$n$={len(d)}, depth={d.max()}',fontsize=7)
    ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal'); ax.axis('off')
fig.tight_layout(); fig.savefig('Fig5.pdf',bbox_inches='tight'); fig.savefig('Fig5.png',dpi=110,bbox_inches='tight')
