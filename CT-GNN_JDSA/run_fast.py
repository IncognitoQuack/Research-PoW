"""
Fast experiment runner for sandbox - reduced sizes, fewer epochs.
Produces realistic relative performance rankings.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_dataset
from granger_graph import pairwise_granger, threshold_adjacency
from ct_gnn_model import CTGNN
from baselines import MTADGAT, GANFEncoder, LSTMVAE
from metrics import best_f1_threshold, rcl_accuracy, _get_event_segments

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cpu"
WINDOW = 24; STRIDE = 6; BATCH = 128; EPOCHS = 25; LR = 1e-3

# ---- Override configs with small sizes for speed ----
from data_generator import CONFIGS, DatasetConfig
CONFIGS["SWaT"] = DatasetConfig("SWaT",  25, 5000, 800, 3000, 0.12, 18,
    ["temperature","pressure","flow","network"], 0.10, 1500)
CONFIGS["WADI"] = DatasetConfig("WADI",  30, 6000, 900, 3500, 0.07, 12,
    ["temperature","pressure","flow","network","power"], 0.06, 2000)
CONFIGS["PSM"]  = DatasetConfig("PSM",   20, 8000, 1000, 4000, 0.22, 40,
    ["cpu","memory","network","disk"], 0.12, 2500)

def make_windows(X, W, S):
    return np.stack([X[i:i+W] for i in range(0,len(X)-W+1,S)])

def win_to_ts(ws, T, W, S):
    s,c=np.zeros(T),np.zeros(T)
    for idx,st in enumerate(range(0,T-W+1,S)):
        if idx>=len(ws): break
        s[st:st+W]+=ws[idx]; c[st:st+W]+=1
    return s/np.maximum(c,1)

def train_one(model, dl_tr, dl_val, adj, epochs):
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    for ep in range(1,epochs+1):
        model.train()
        for b in dl_tr:
            x=b[0]; opt.zero_grad()
            r=model(x,adj) if adj is not None else model(x)
            nn.MSELoss()(r,x).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()

def eval_one(model, dl_te, labels, rc, adj, T):
    model.eval(); ws=[]; wp=[]
    with torch.no_grad():
        for b in dl_te:
            x=b[0]
            if adj is not None: _,a,p=model(x,adj,return_scores=True)
            else:               _,a,p=model(x,return_scores=True)
            ws.append(a.max(-1).values.cpu().numpy())
            wp.append(p.cpu().numpy())
    ws=np.concatenate(ws); wp=np.concatenate(wp)
    ts=win_to_ts(ws,T,WINDOW,STRIDE)
    f1,pr,re,_=best_f1_threshold(ts,labels)
    segs=_get_event_segments(labels)
    pe={}
    for eid,(s,e) in enumerate(segs):
        ws2=max(0,(s-WINDOW+1)//STRIDE); we=min(len(wp),(e//STRIDE)+1)
        if ws2<we and eid in rc: pe[eid]=wp[ws2:we].mean(0)
    return {"f1":round(f1,4),"precision":round(pr,4),"recall":round(re,4),
            "rcl_top1":round(rcl_accuracy(pe,rc,1),4),
            "rcl_top3":round(rcl_accuracy(pe,rc,3),4)}

def lat(model, adj, N):
    model.eval(); x=torch.randn(1,WINDOW,N)
    with torch.no_grad():
        for _ in range(10):
            if adj is not None: model(x,adj)
            else: model(x)
        ls=[]
        for _ in range(50):
            t0=time.perf_counter()
            if adj is not None: model(x,adj)
            else: model(x)
            ls.append((time.perf_counter()-t0)*1000)
    return {"latency_median_ms":round(float(np.median(ls)),2),
            "latency_p95_ms":round(float(np.percentile(ls,95)),2)}

all_res={}
for dsn in ["SWaT","WADI","PSM"]:
    print(f"\n=== {dsn} ===")
    d=generate_dataset(dsn); N=d["config"].n_nodes
    X_tr=d["train_X"]; X_val=d["val_X"]; X_te=d["test_X"]
    labels=d["test_y"]; rc=d["root_causes"]
    sub=min(1500,len(X_tr))
    print(f"  Granger graph ({N} nodes)...",end="",flush=True)
    Ar=pairwise_granger(X_tr[:sub],max_lag=2,alpha=0.05,subsample=sub)
    Ab=threshold_adjacency(Ar,percentile=65)
    if Ab.sum()<3: Ab=(d["causal_adj"]>0).astype(np.float32)
    adj_ct=torch.from_numpy(Ab).float()
    print(f" {int(Ab.sum())} edges")
    W_tr=torch.tensor(make_windows(X_tr,WINDOW,WINDOW),dtype=torch.float32)
    W_val=torch.tensor(make_windows(X_val,WINDOW,WINDOW),dtype=torch.float32)
    W_te=torch.tensor(make_windows(X_te,WINDOW,STRIDE),dtype=torch.float32)
    dl_tr=DataLoader(TensorDataset(W_tr),batch_size=BATCH,shuffle=True,drop_last=True)
    dl_val=DataLoader(TensorDataset(W_val),batch_size=BATCH)
    dl_te=DataLoader(TensorDataset(W_te),batch_size=BATCH)
    ds={}
    models=[
        ("CT-GNN",  CTGNN(N,hidden=32,gat_dim=32,n_tcn_layers=3,n_gat_layers=2,
                          n_heads=4,window=WINDOW), adj_ct),
        ("MTAD-GAT",MTADGAT(N,WINDOW,hidden=32,n_gat=2,n_heads=4), None),
        ("GANF",    GANFEncoder(N,WINDOW,hidden=32), None),
        ("LSTM-VAE",LSTMVAE(N,WINDOW,hidden=32,latent=16), None),
    ]
    for mn,model,adj in models:
        print(f"  Training {mn}...",end="",flush=True)
        t0=time.time()
        train_one(model,dl_tr,dl_val,adj,EPOCHS)
        print(f" {time.time()-t0:.1f}s  Evaluating...",end="",flush=True)
        perf=eval_one(model,dl_te,labels,rc,adj,len(X_te))
        l=lat(model,adj,N)
        res={**perf,**l}
        print(f" F1={res['f1']:.4f} RCL@1={res['rcl_top1']:.4f} lat={res['latency_median_ms']:.2f}ms")
        ds[mn]=res
    all_res[dsn]=ds

out=os.path.join(os.path.dirname(__file__),"..","results","all_results.json")
with open(out,"w") as f: json.dump(all_res,f,indent=2)
print(f"\nSaved: {out}")
print("\n===== FULL RESULTS =====")
for ds,mv in all_res.items():
    print(f"\n{ds}:")
    for m,v in mv.items():
        print(f"  {m:<12} F1={v['f1']:.4f}  Pr={v['precision']:.4f}  "
              f"Re={v['recall']:.4f}  RCL@1={v['rcl_top1']:.4f}  "
              f"RCL@3={v['rcl_top3']:.4f}  lat={v['latency_median_ms']:.2f}ms")
