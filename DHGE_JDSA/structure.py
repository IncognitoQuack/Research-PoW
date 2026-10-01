import numpy as np, sys, json, itertools, random
sys.path.insert(0,'.'); from data import *
ROOT='/home/claude/STS-NN/dataset'
def graph_edges(ds):
    # all distinct parent links (multi-parent rows kept) from the full-horizon file
    d={}
    for l in open(f'{ROOT}/{ds}/data.strnn.vol5000.f0.et2160.txt',errors='ignore'):
        a=l.rstrip('\r\n').split('\t'); i=int(a[1])
        ps={int(p) for p in a[2].split() if p!='None' and int(p)<i}
        if i>0 and not ps: ps={0}
        d.setdefault(a[0],{})[i]=ps
    return d
def apsp(n,adj):
    D=np.full((n,n),np.inf)
    for s in range(n):
        D[s,s]=0; q=[s]
        for u in q:
            for v in adj[u]:
                if D[s,v]==np.inf: D[s,v]=D[s,u]+1; q.append(v)
    return D
def delta(D, samples=20000, rng=random.Random(0)):
    n=len(D)
    if n<4: return 0.0
    best=0.0
    quads = itertools.combinations(range(n),4) if n<=30 else (rng.sample(range(n),4) for _ in range(samples))
    for x,y,z,w in quads:
        s=sorted([D[x,y]+D[z,w],D[x,z]+D[y,w],D[x,w]+D[y,z]])
        best=max(best,(s[2]-s[1])/2)
    return best
out={}
for ds in ['twitter15','twitter16']:
    E=graph_edges(ds); G,_,_=load_fold(ds,0)
    multi=0; tot=0; deltas=[]; rel=[]; depth={c:[] for c in range(4)}; width={c:[] for c in range(4)}; diam=[]
    for e,nodes in E.items():
        n=len(nodes); adj=[set() for _ in range(n)]
        for i,ps in nodes.items():
            tot+=1; multi+= len(ps)>1
            for p in ps: adj[i].add(p); adj[p].add(i)
        D=apsp(n,adj); dl=delta(D); dm=D[np.isfinite(D)].max()
        deltas.append(dl); diam.append(dm); rel.append(2*dl/dm if dm>0 else 0)
        g=G[e]; d=np.zeros(len(g['par']),int)
        for i in range(1,len(g['par'])): d[i]=d[g['par'][i]]+1
        depth[g['y']].append(d.max()); width[g['y']].append(np.bincount(d).max())
    deltas=np.array(deltas); rel=np.array(rel)
    out[ds]=dict(n=len(E), multi_parent_frac=multi/tot, delta_mean=float(deltas.mean()), delta_max=float(deltas.max()),
        frac_delta0=float((deltas==0).mean()), frac_delta_le1=float((deltas<=1).mean()), rel_delta_mean=float(rel.mean()),
        diam_mean=float(np.mean(diam)),
        depth={c:(float(np.mean(v)),float(np.std(v))) for c,v in depth.items()},
        width={c:(float(np.mean(v)),float(np.std(v))) for c,v in width.items()})
    print(ds,json.dumps(out[ds],indent=0))
json.dump(out,open('structure.json','w'),indent=1)
