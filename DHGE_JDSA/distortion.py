import numpy as np, torch, random, json, sys
sys.path.insert(0,'.'); from data import load_fold
torch.set_num_threads(1)
def apsp(par):
    n=len(par); adj=[[] for _ in range(n)]
    for i in range(1,n): adj[i].append(par[i]); adj[par[i]].append(i)
    D=np.zeros((n,n))
    for s in range(n):
        seen=[-1]*n; seen[s]=0; q=[s]
        for u in q:
            for v in adj[u]:
                if seen[v]<0: seen[v]=seen[u]+1; q.append(v)
        D[s]=seen
    return D
def pdist_ball(x):
    x2=(x*x).sum(-1,keepdim=True); diff=((x[:,None]-x[None])**2).sum(-1)
    arg=1+2*diff/((1-x2)*(1-x2.T)).clamp(min=1e-9)
    return torch.acosh(arg.clamp(min=1+1e-7))
def fit(D, geom, dim=2, steps=400, seed=0):
    torch.manual_seed(seed); n=len(D); Dt=torch.tensor(D,dtype=torch.float32)
    mask=torch.triu(torch.ones(n,n),1)>0
    v=torch.nn.Parameter(0.1*torch.randn(n,dim)); s=torch.nn.Parameter(torch.tensor(0.))
    opt=torch.optim.Adam([v,s],lr=0.05)
    def dist():
        if geom=='hyp':
            nv=v.norm(dim=-1,keepdim=True).clamp(min=1e-7); x=torch.tanh(nv).clamp(max=1-1e-5)*v/nv
            return pdist_ball(x)*torch.exp(s)   # global scale ~ curvature choice
        return torch.cdist(v,v)
    for _ in range(steps):
        opt.zero_grad(); d=dist(); loss=(((d-Dt)/Dt.clamp(min=1))**2)[mask].mean(); loss.backward(); opt.step()
    with torch.no_grad():
        d=dist()[mask]; g=Dt[mask]; return float(((d-g).abs()/g).mean())
if __name__ == "__main__":
    res={}
    for ds in ['twitter15','twitter16']:
        G,_,_=load_fold(ds,0); rng=random.Random(0)
        es=[e for e in G if len(G[e]['par'])>=5]; es=rng.sample(es,150)
        r={'hyp2':[], 'euc2':[], 'euc8':[]}
        for e in es:
            D=apsp(G[e]['par'])
            r['hyp2'].append(fit(D,'hyp',2)); r['euc2'].append(fit(D,'euc',2)); r['euc8'].append(fit(D,'euc',8))
        res[ds]={k:(float(np.mean(v)),float(np.std(v))) for k,v in r.items()}
        from scipy.stats import wilcoxon
        res[ds]['p_hyp2_vs_euc2']=float(wilcoxon(r['hyp2'],r['euc2']).pvalue)
        res[ds]['p_hyp2_vs_euc8']=float(wilcoxon(r['hyp2'],r['euc8']).pvalue)
        res[ds]['n']=len(es); print(ds,res[ds],flush=True)
    json.dump(res,open('distortion.json','w'),indent=1)
