import json, collections, numpy as np, os
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42,'axes.titlesize':8,'legend.fontsize':7})
R = [json.loads(l) for l in open('results.jsonl')]
MAIN = ['BiGCN','DDGCN','DynGCN','CGNKP','HGCN','D-HGE']; ABL = ['D-HGE','Euclidean','Static','NoCurvAttn','NoRiskScore']
DS = ['twitter15','twitter16']; DSN = {'twitter15':'Twitter15','twitter16':'Twitter16'}
MET = ['acc','f1','auc','eda20','mdl']
def cell(ds, m, seed_filter=None):
    return [r for r in R if r['ds']==ds and r['model']==m and (seed_filter is None or r['seed'] in seed_filter)]
S = {}
for ds in DS:
    for m in set(MAIN+ABL):
        rs = cell(ds, m)
        if not rs: continue
        d = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1)) if len(rs)>1 else 0.0) for k in MET}
        d['n'] = len(rs); d['acc_at'] = {T: float(np.mean([r['acc_at'][T] for r in rs])) for T in rs[0]['acc_at']}
        d['detected'] = float(np.mean([r['detected_frac'] for r in rs]))
        d['per_class_f1'] = np.mean([r['per_class_f1'] for r in rs], 0).tolist()
        S[f'{ds}|{m}'] = d
# exact McNemar on pooled per-claim predictions (each claim predicted once per model)
from scipy.stats import binomtest
P = {}
def pooled(ds, m, key='correct'):
    d = {}
    for r in cell(ds, m):
        for e, c in zip(r['ids'], r[key]): d[e] = c
    return d
for ds in DS:
    A = pooled(ds, 'D-HGE'); A20 = pooled(ds, 'D-HGE', 'correct20')
    for m in MAIN + ABL[1:]:
        if m == 'D-HGE' or not cell(ds, m): continue
        for key, AA in [('acc', A), ('eda20', A20)]:
            B = pooled(ds, m, 'correct' if key == 'acc' else 'correct20'); ks = sorted(set(AA) & set(B))
            b01 = sum(1 for k in ks if AA[k] == 1 and B[k] == 0); b10 = sum(1 for k in ks if AA[k] == 0 and B[k] == 1)
            p = float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
            P[f'{ds}|{m}|{key}'] = dict(p=p, dhge_only=b01, other_only=b10, n=len(ks),
                                        diff=(sum(AA[k] for k in ks) - sum(B[k] for k in ks)) / max(1, len(ks)))
json.dump(dict(S=S, P=P), open('summary.json','w'), indent=1)
# ------------- figures -------------
COL = {'BiGCN':'#ff7f0e','DDGCN':'#2ca02c','DynGCN':'#d62728','CGNKP':'#9467bd','HGCN':'#8c564b','D-HGE':'#1f77b4'}
def bars(metrics, labels, fname, ylims=None):
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.5))
    x = np.arange(len(DS)); w = 0.13; off = (np.arange(len(MAIN)) - (len(MAIN)-1)/2) * w
    for ax, k, lab in zip(axes, metrics, labels):
        for i, m in enumerate(MAIN):
            mu = [S[f'{ds}|{m}'][k][0] for ds in DS]; sd = [S[f'{ds}|{m}'][k][1] for ds in DS]
            b = ax.bar(x+off[i], mu, w*0.92, yerr=sd, capsize=1.5, color=COL[m], label=m, zorder=3, error_kw=dict(lw=0.6))
            if m == 'D-HGE':
                for bb in b: bb.set_hatch('//'); bb.set_edgecolor('#0d4a8a')
        ax.set_xticks(x); ax.set_xticklabels([DSN[d] for d in DS]); ax.set_ylabel(lab)
        ax.yaxis.grid(True, ls='--', alpha=0.5, zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ylims: ax.set_ylim(*ylims[k])
    axes[0].legend(ncol=3, loc='upper left', fontsize=6, framealpha=0.85)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
if all(f'{ds}|{m}' in S for ds in DS for m in MAIN):
    bars(['acc','f1'], ['Accuracy','Macro-F1'], 'figs/Fig2.pdf', {'acc':(0.3,0.95),'f1':(0.3,0.95)})
    bars(['eda20','mdl'], ['EDA@20%','MDL (min, lower is better)'], 'figs/Fig3.pdf', {'eda20':(0.3,0.95),'mdl':(0,2300)})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, ds in zip(axes, DS):
        for m in MAIN:
            a = S[f'{ds}|{m}']['acc_at']; Ts = sorted(a, key=int)
            ax.plot(range(len(Ts)), [a[T] for T in Ts], marker='o', ms=3, lw=1.6 if m=='D-HGE' else 1.0, color=COL[m], label=m, ls='-' if m=='D-HGE' else '--')
        ax.set_xticks(range(len(Ts))); ax.set_xticklabels(Ts); ax.set_xlabel('Observation deadline (min)'); ax.set_ylabel('Accuracy')
        ax.set_title(DSN[ds]); ax.yaxis.grid(True, ls='--', alpha=0.5); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].legend(ncol=2, fontsize=6); fig.tight_layout(); fig.savefig('figs/Fig6.pdf', bbox_inches='tight'); plt.close(fig)
if all(f'twitter16|{m}' in S for m in ABL):
    fig, ax = plt.subplots(figsize=(7.0, 2.4)); x = np.arange(len(ABL)); w = 0.2
    names = ['Full D-HGE', 'Euclidean', 'Static', 'NoCurvAttn', 'NoRiskScore']
    for j, (k, c, lab) in enumerate([('acc','#1f77b4','Accuracy'),('f1','#ff7f0e','Macro-F1'),('auc','#2ca02c','AUROC'),('eda20','#9467bd','EDA@20%')]):
        mu = [S[f'twitter16|{m}'][k][0] for m in ABL]; sd = [S[f'twitter16|{m}'][k][1] for m in ABL]
        ax.bar(x+(j-1.5)*w, mu, w*0.92, yerr=sd, capsize=1.5, color=c, label=lab, zorder=3, error_kw=dict(lw=0.6))
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.3, 1.0); ax.legend(ncol=4, fontsize=6.5, loc='upper right')
    ax.yaxis.grid(True, ls='--', alpha=0.5, zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig('figs/Fig4.pdf', bbox_inches='tight'); plt.close(fig)
print('summary ok', len(R))
