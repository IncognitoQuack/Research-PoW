"""
NeuroSym-ZTA: Phase 2 — Complete Experimental Pipeline
=======================================================
Paper  : Neuro-Symbolic Approach for Explainable Authentication and
         Access Control in Zero-Trust Network Architectures
Journal: Journal of Information Security and Applications (JISA), Elsevier
Dataset: NSL-KDD  (KDDTrain+.txt  |  KDDTest+.txt)  — auto-downloaded

Run    : python neurosym_zta_phase2.py           # full  (30 epochs)
         python neurosym_zta_phase2.py --quick   # smoke (8 epochs)

Outputs → ./results/
"""
# ── stdlib ────────────────────────────────────────────────────────────────
import os, sys, time, json, warnings, argparse
warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request

# ── reproducibility ───────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# 0.  CLI + CONFIG
# ═══════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args, _ = parser.parse_known_args()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "./results";  os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_DIR    = "./data";     os.makedirs(DATA_DIR,    exist_ok=True)

SEQ_LEN    = 10
STRIDE     = 5
BATCH_SIZE = 256
EPOCHS     = 8 if args.quick else 30
HIDDEN     = 64
N_RULES    = 5
N_CLS      = 3          # 0=ALLOW  1=CHALLENGE  2=DENY
LR         = 1e-3
N_SHAP     = 30 if args.quick else 120   # samples for attribution
ZTA_NAMES  = ["ALLOW", "CHALLENGE", "DENY"]

print("=" * 65)
print("NeuroSym-ZTA  |  Phase 2")
print(f"Device: {DEVICE}   Mode: {'QUICK' if args.quick else 'FULL'}")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
TRAIN_F = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_F  = os.path.join(DATA_DIR, "KDDTest+.txt")
_URLS = {
    "train": ["https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"],
    "test":  ["https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"],
}

def ensure_data():
    for split, path in [("train", TRAIN_F), ("test", TEST_F)]:
        if os.path.exists(path): continue
        print(f"Downloading {split} set …")
        ok = False
        for url in _URLS[split]:
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  Saved ({os.path.getsize(path)//1024:,} KB)")
                ok = True; break
            except Exception as e:
                print(f"  {e}")
        if not ok:
            print("Manual download: https://www.unb.ca/cic/datasets/nsl.html")
            return False
    return True

# ═══════════════════════════════════════════════════════════════════════════
# 2.  SCHEMA
# ═══════════════════════════════════════════════════════════════════════════
_COLS = [
    "duration","protocol_type","service","flag",
    "src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty",
]
CAT = ["protocol_type","service","flag"]
NUM = [c for c in _COLS[:-2] if c not in CAT]   # 38 numeric
ALL_FEAT = NUM + CAT                              # 41 total

# ═══════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESSING + ZTA LABEL MAPPING
# ═══════════════════════════════════════════════════════════════════════════
_DOS   = {"neptune","smurf","pod","teardrop","back","land","processtable",
          "udpstorm","apache2","mailbomb","sendmail","named",
          "snmpgetattack","snmpguess","worm"}
_PROBE = {"ipsweep","nmap","portsweep","satan","saint","mscan"}

def _zta(lbl: str) -> int:
    l = lbl.strip().lower()
    if l == "normal":                    return 0  # ALLOW
    if l in _DOS or l in _PROBE:         return 1  # CHALLENGE
    return 2                                        # DENY  (R2L / U2R)

def load_preprocess():
    print("\nLoading NSL-KDD …")
    df_tr = pd.read_csv(TRAIN_F, header=None, names=_COLS)
    df_te = pd.read_csv(TEST_F,  header=None, names=_COLS)
    print(f"  Train {len(df_tr):,}  |  Test {len(df_te):,}")

    df_tr["zta"] = df_tr["label"].apply(_zta)
    df_te["zta"] = df_te["label"].apply(_zta)

    # label-encode categoricals
    les = {}
    for c in CAT:
        le = LabelEncoder()
        df_tr[c] = le.fit_transform(df_tr[c].astype(str))
        df_te[c] = df_te[c].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0)
        les[c] = le

    X_tr = df_tr[ALL_FEAT].fillna(0).astype(float).values
    X_te = df_te[ALL_FEAT].fillna(0).astype(float).values
    y_tr = df_tr["zta"].values.astype(int)
    y_te = df_te["zta"].values.astype(int)

    sc = RobustScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    def _dist(y, tag):
        bc = np.bincount(y, minlength=3)
        print(f"  {tag}: ALLOW {bc[0]:,}  CHALLENGE {bc[1]:,}  DENY {bc[2]:,}")
    _dist(y_tr, "Train"); _dist(y_te, "Test ")

    return X_tr, X_te, y_tr, y_te, sc, df_tr, df_te

# ═══════════════════════════════════════════════════════════════════════════
# 4.  SYMBOLIC RULE ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════
RULE_NAMES = [
    "R1: Auth-Failure Rate",
    "R2: Priv-Escalation Signal",
    "R3: Connection Error Rate",
    "R4: Service Diversity",
    "R5: Host-Compromise Flag",
]

def compute_rules(df: pd.DataFrame) -> np.ndarray:
    R = np.zeros((len(df), N_RULES), dtype=np.float32)
    R[:,0] = np.clip(df["num_failed_logins"].fillna(0).values / 5.0,  0, 1)
    R[:,1] = np.clip(
        (df["root_shell"].fillna(0).values +
         df["su_attempted"].fillna(0).values) / 2.0, 0, 1)
    R[:,2] = np.clip(df["serror_rate"].fillna(0).values,               0, 1)
    R[:,3] = np.clip(df["diff_srv_rate"].fillna(0).values,             0, 1)
    R[:,4] = np.clip(df["num_compromised"].fillna(0).values / 10.0,    0, 1)
    return R

# ═══════════════════════════════════════════════════════════════════════════
# 5.  SEQUENCE CONSTRUCTION
#     FIX: shuffle first → centre-element label → realistic class distribution
# ═══════════════════════════════════════════════════════════════════════════
def make_sequences(X, R, y, seq_len=SEQ_LEN, stride=STRIDE):
    """
    Sliding-window session construction.
    Data is shuffled BEFORE windowing so that ALLOW / CHALLENGE / DENY
    records are interleaved, preventing artificial label collapse.
    Label = the record at the centre of the window (representative sample).
    """
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    X, R, y = X[perm], R[perm], y[perm]

    mid = seq_len // 2
    Xs, Rs, ys = [], [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        Xs.append(X[i: i + seq_len])
        Rs.append(R[i: i + seq_len].mean(0))
        ys.append(int(y[i + mid]))          # centre label
    return (np.array(Xs, np.float32),
            np.array(Rs, np.float32),
            np.array(ys, np.int64))

# ═══════════════════════════════════════════════════════════════════════════
# 6.  MODELS
# ═══════════════════════════════════════════════════════════════════════════
class SymbolicRuleLayer(nn.Module):
    """Differentiable ZTA policy engine — rule weights are LEARNED."""
    def __init__(self):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(N_RULES))
        self.head = nn.Linear(1, N_CLS)

    def forward(self, r):           # r: (B, N_RULES)
        score = torch.sigmoid((self.w * r).sum(1, keepdim=True))   # (B,1)
        return self.head(score), score

    def weights(self):
        return self.w.detach().cpu().numpy()


class NeuroSymZTA(nn.Module):
    """
    Proposed model.
    Neural path  : BiGRU (2-layer, bidirectional) + temporal attention
    Symbolic path: Differentiable 5-rule ZTA policy layer
    Fusion        : Entropy-adaptive weighting
    """
    def __init__(self, in_dim, hidden=HIDDEN, drop=0.30):
        super().__init__()
        self.n_cls = N_CLS
        self.gru   = nn.GRU(in_dim, hidden, 2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.attn  = nn.Linear(hidden*2, 1)
        self.drop  = nn.Dropout(drop)
        self.nhead = nn.Linear(hidden*2, N_CLS)
        self.sym   = SymbolicRuleLayer()
        self.gate  = nn.Parameter(torch.tensor(0.6))

    def forward(self, x, r):
        g, _ = self.gru(x)
        a    = torch.softmax(self.attn(g), 1)
        c    = self.drop((a * g).sum(1))
        n_l  = self.nhead(c);  n_p = torch.softmax(n_l, 1)

        s_l, s_sc = self.sym(r); s_p = torch.softmax(s_l, 1)

        H    = -(n_p * torch.log(n_p + 1e-8)).sum(1, keepdim=True)
        Hmax = torch.log(torch.tensor(float(self.n_cls), device=H.device))
        phi  = 1.0 - H / Hmax
        alp  = torch.sigmoid(self.gate) * phi

        return alp * n_p + (1 - alp) * s_p, n_p, s_sc, alp


class StandaloneGRU(nn.Module):
    """Baseline B1 — neural only."""
    def __init__(self, in_dim, hidden=HIDDEN, drop=0.30):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, 2, batch_first=True,
                           bidirectional=True, dropout=drop)
        self.attn = nn.Linear(hidden*2, 1)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden*2, N_CLS)

    def forward(self, x, _r=None):
        g, _ = self.gru(x)
        a    = torch.softmax(self.attn(g), 1)
        c    = self.drop((a * g).sum(1))
        return torch.softmax(self.head(c), 1)

# ═══════════════════════════════════════════════════════════════════════════
# 7.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════
class _DS(Dataset):
    def __init__(self, X, R, y):
        self.X=torch.FloatTensor(X); self.R=torch.FloatTensor(R); self.y=torch.LongTensor(y)
    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.R[i], self.y[i]

def train_model(model, Xtr, Rtr, ytr, Xvl, Rvl, yvl, name=""):
    counts = np.bincount(ytr, minlength=N_CLS).astype(float)
    w = torch.FloatTensor(1.0 / (counts + 1)).to(DEVICE)
    w /= w.sum()
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt     = optim.Adam(model.parameters(), lr=LR)
    sched   = optim.lr_scheduler.StepLR(opt, max(1, EPOCHS//3), gamma=0.5)
    loader  = DataLoader(_DS(Xtr, Rtr, ytr), BATCH_SIZE, shuffle=True, num_workers=0)
    model.to(DEVICE)

    hist = {"loss":[], "val_acc":[], "val_f1":[]}
    best_f1, best_sd = -1.0, None

    for ep in range(1, EPOCHS+1):
        model.train(); ep_loss = 0.0
        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(DEVICE), rb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb, rb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            Xv=torch.FloatTensor(Xvl).to(DEVICE); Rv=torch.FloatTensor(Rvl).to(DEVICE)
            out_v = model(Xv, Rv)
            pv = (out_v[0] if isinstance(out_v,tuple) else out_v).argmax(1).cpu().numpy()
        va = accuracy_score(yvl, pv)
        vf = f1_score(yvl, pv, average="macro", zero_division=0)
        hist["loss"].append(ep_loss/len(loader))
        hist["val_acc"].append(va); hist["val_f1"].append(vf)
        if vf > best_f1:
            best_f1 = vf
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        step = max(1, EPOCHS//5)
        if ep % step == 0:
            print(f"  [{name}] ep {ep:02d}  loss={ep_loss/len(loader):.4f}  "
                  f"val_acc={va:.4f}  val_f1={vf:.4f}")

    if best_sd: model.load_state_dict(best_sd)
    return model, hist

def eval_model(model, Xte, Rte, yte):
    model.eval()
    with torch.no_grad():
        Xt=torch.FloatTensor(Xte).to(DEVICE); Rt=torch.FloatTensor(Rte).to(DEVICE)
        out = model(Xt, Rt)
        probs = (out[0] if isinstance(out,tuple) else out).cpu().numpy()
    preds = probs.argmax(1)
    yb = label_binarize(yte, classes=[0,1,2])
    try:
        # Only compute AUC if all classes present
        if len(np.unique(yte)) == 3:
            auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")
    return dict(Accuracy=accuracy_score(yte,preds),
                F1=f1_score(yte,preds,average="macro",zero_division=0),
                Precision=precision_score(yte,preds,average="macro",zero_division=0),
                Recall=recall_score(yte,preds,average="macro",zero_division=0),
                AUC=auc, preds=preds, probs=probs)

# ═══════════════════════════════════════════════════════════════════════════
# 8.  BASELINES
# ═══════════════════════════════════════════════════════════════════════════
def eval_rf(Xtr_s, Rtr, ytr, Xte_s, Rte, yte):
    Xfl_tr = np.hstack([Xtr_s.mean(1), Rtr])
    Xfl_te = np.hstack([Xte_s.mean(1), Rte])
    rf = RandomForestClassifier(200, class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xfl_tr, ytr)
    probs = rf.predict_proba(Xfl_te)
    preds = probs.argmax(1)
    yb = label_binarize(yte, classes=[0,1,2])
    try:
        auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr") \
              if len(np.unique(yte))==3 else float("nan")
    except Exception: auc = float("nan")
    return rf, dict(Accuracy=accuracy_score(yte,preds),
                    F1=f1_score(yte,preds,average="macro",zero_division=0),
                    Precision=precision_score(yte,preds,average="macro",zero_division=0),
                    Recall=recall_score(yte,preds,average="macro",zero_division=0),
                    AUC=auc, preds=preds, probs=probs), Xfl_tr, Xfl_te

def eval_rule_engine(Rte, yte, lo=0.25, hi=0.55):
    risk  = Rte.mean(1)
    preds = np.zeros(len(risk), int)
    preds[risk >= lo] = 1
    preds[risk >= hi] = 2
    probs = np.column_stack([np.clip(1-risk,0,1),
                             np.clip(risk*(1-risk)*4,0,1),
                             np.clip(risk,0,1)])
    probs /= probs.sum(1, keepdims=True) + 1e-8
    yb = label_binarize(yte, classes=[0,1,2])
    try:
        auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr") \
              if len(np.unique(yte))==3 else float("nan")
    except Exception: auc = float("nan")
    return dict(Accuracy=accuracy_score(yte,preds),
                F1=f1_score(yte,preds,average="macro",zero_division=0),
                Precision=precision_score(yte,preds,average="macro",zero_division=0),
                Recall=recall_score(yte,preds,average="macro",zero_division=0),
                AUC=auc, preds=preds, probs=probs)

# static-fusion variant of NeuroSymZTA
def eval_static_fusion(model, Xte, Rte, yte):
    model.eval()
    with torch.no_grad():
        Xt=torch.FloatTensor(Xte).to(DEVICE); Rt=torch.FloatTensor(Rte).to(DEVICE)
        _, np_, ss, _ = model(Xt, Rt)
        sp = torch.softmax(model.sym.head(ss), 1)
        probs = (0.5*np_ + 0.5*sp).cpu().numpy()
    preds = probs.argmax(1)
    yb = label_binarize(yte, classes=[0,1,2])
    try:
        auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr") \
              if len(np.unique(yte))==3 else float("nan")
    except Exception: auc = float("nan")
    return dict(Accuracy=accuracy_score(yte,preds),
                F1=f1_score(yte,preds,average="macro",zero_division=0),
                Precision=precision_score(yte,preds,average="macro",zero_division=0),
                Recall=recall_score(yte,preds,average="macro",zero_division=0),
                AUC=auc, preds=preds, probs=probs)

# ═══════════════════════════════════════════════════════════════════════════
# 9.  SHAP  — gradient × input for neural path  (no KernelSHAP compat issues)
#            TreeSHAP for RF  (instant)
# ═══════════════════════════════════════════════════════════════════════════
def shap_gradient_input(model, Xte_s, n=N_SHAP):
    """
    Gradient × Input attribution for the NeuroSym-ZTA neural path.
    Attributes the DENY-class probability to input features.
    Returns mean absolute attribution per feature (averaged over seq dim).
    """
    model.eval()
    idx = np.random.RandomState(SEED).choice(len(Xte_s), min(n,len(Xte_s)), replace=False)
    Xt  = torch.FloatTensor(Xte_s[idx]).to(DEVICE).requires_grad_(True)
    Rt  = torch.zeros(len(Xt), N_RULES, device=DEVICE)

    fused, n_prob, _, _ = model(Xt, Rt)
    deny = n_prob[:, 2].sum()          # DENY column, neural path
    deny.backward()

    attr = (Xt.grad * Xt).abs()        # (n, seq_len, feat_dim)
    mean_attr = attr.mean(0).mean(0).detach().cpu().numpy()   # (feat_dim,)
    return mean_attr

def shap_tree(rf, Xfl_te, n=N_SHAP):
    """TreeSHAP on RF — mean |SHAP| per feature (DENY class = index 2)."""
    idx = np.random.RandomState(SEED).choice(len(Xfl_te), min(n,len(Xfl_te)), replace=False)
    ex  = shap.TreeExplainer(rf)
    sv  = ex.shap_values(Xfl_te[idx])     # list[C] of (n, F)
    return np.abs(sv[2]).mean(0)           # DENY class

# ═══════════════════════════════════════════════════════════════════════════
# 10.  PRIVILEGE-ESCALATION TABLE
# ═══════════════════════════════════════════════════════════════════════════
def priv_esc_table(yte, preds_dict):
    rows = []
    for name, preds in preds_dict.items():
        rows.append(dict(Method=name,
            DENY_Prec =round(precision_score(yte,preds,labels=[2],average="macro",zero_division=0),4),
            DENY_Recall=round(recall_score(   yte,preds,labels=[2],average="macro",zero_division=0),4),
            DENY_F1   =round(f1_score(        yte,preds,labels=[2],average="macro",zero_division=0),4)))
    df = pd.DataFrame(rows)
    print("\nPrivilege Escalation Resistance (DENY class):")
    print(df.to_string(index=False))
    df.to_csv(f"{RESULTS_DIR}/priv_escalation_table.csv", index=False)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 11.  LATENCY
# ═══════════════════════════════════════════════════════════════════════════
def latency_bench(model, Xte_s, Rte_s):
    model.eval(); out = {}
    for n in (100, 500, 1000, 2000):
        n = min(n, len(Xte_s))
        Xt=torch.FloatTensor(Xte_s[:n]).to(DEVICE)
        Rt=torch.FloatTensor(Rte_s[:n]).to(DEVICE)
        with torch.no_grad():
            for _ in range(3): model(Xt, Rt)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10): model(Xt, Rt)
        ms = (time.perf_counter()-t0)/10*1000
        out[n] = round(ms, 2)
        print(f"  {n:5d} sessions → {ms:.2f} ms")
    return out

# ═══════════════════════════════════════════════════════════════════════════
# 12.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.size":11,"axes.labelsize":11,"axes.titlesize":11,
                     "xtick.labelsize":10,"ytick.labelsize":10,
                     "legend.fontsize":9,"savefig.dpi":300,"savefig.bbox":"tight"})
PAL = {
    "NeuroSym-ZTA (Ours)":   "#1f4e79",
    "Standalone BiGRU":       "#2e75b6",
    "No Symbolic Layer":      "#5a9fd4",
    "Static Fusion (α=0.5)":  "#7030a0",
    "Random Forest":          "#ed7d31",
    "Rule Engine Only":       "#70ad47",
}

def _save(fig, name):
    fig.savefig(f"{RESULTS_DIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{RESULTS_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  ✓ {name}")

def fig1_arch():
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
    def box(x,y,w,h,fc,txt,tc="white",fs=8.5):
        ax.add_patch(plt.Rectangle((x,y),w,h,fc=fc,ec="#333",lw=1.2,zorder=2))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
                fontweight="bold",color=tc,zorder=3,multialignment="center")
    def arr(x1,y1,x2,y2,rad=0.0):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->",color="#444",lw=1.4,
                                   connectionstyle=f"arc3,rad={rad}"),zorder=4)
    box(0.2,4.85,1.5,0.7,"#bdd7ee","Auth Logs\n(NSL-KDD)","#1f1f1f")
    box(0.2,3.75,1.5,0.75,"#9dc3e6","Layer 1\nPreprocessing\n& Features","#1f1f1f")
    box(0.2,2.50,1.5,0.90,"#2e75b6","Layer 2\nBiGRU + Attention\nBehavioral Encoder")
    box(0.2,1.10,1.5,1.00,"#70ad47","Layer 3\nDifferentiable\nSymbolic Rules\n(5 ZTA Policies)")
    box(3.85,2.50,1.9,0.90,"#ed7d31","Layer 4\nAdaptive Entropy\nFusion\nα·neural+(1-α)·sym")
    box(6.90,3.50,2.75,0.90,"#7030a0","Layer 5\nDual Explainability\nGrad×Input + Rule Trace")
    box(6.90,2.00,2.75,1.00,"#c55a11","Layer 6 — Access Decision\nALLOW  /  CHALLENGE  /  DENY")
    arr(0.95,4.85,0.95,4.50); arr(0.95,3.75,0.95,3.40)
    arr(1.70,2.95,3.85,2.95); arr(1.70,1.60,3.85,2.60,rad=-0.3)
    arr(5.75,2.95,6.90,3.95); arr(5.75,2.95,6.90,2.50)
    ax.set_title("Fig. 1 — NeuroSym-ZTA System Architecture",fontsize=11,fontweight="bold",pad=10)
    _save(fig,"fig1_architecture")

def fig2_confusion(yte, preds):
    cm  = confusion_matrix(yte, preds, labels=[0,1,2])
    pct = cm.astype(float) / (cm.sum(1,keepdims=True)+1e-8)*100
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04).set_label("Rate (%)")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{pct[i,j]:.1f}%\n({cm[i,j]})",ha="center",va="center",
                    fontsize=9,color="white" if pct[i,j]>60 else "black")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(ZTA_NAMES); ax.set_yticklabels(ZTA_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Fig. 2 — Confusion Matrix\nNeuroSym-ZTA (Ours) on NSL-KDD Test Set",fontweight="bold")
    _save(fig,"fig2_confusion")

def fig3_roc(res_dict, yte):
    from sklearn.metrics import roc_curve, auc as sk_auc
    yb = label_binarize(yte, classes=[0,1,2])
    fig, ax = plt.subplots(figsize=(6,5))
    for name, res in res_dict.items():
        probs = res.get("probs")
        if probs is None: continue
        try:
            fpr,tpr,_ = roc_curve(yb.ravel(), probs.ravel())
            ra = sk_auc(fpr,tpr)
            ax.plot(fpr,tpr,lw=2.5 if "Ours" in name else 1.5,
                    ls="-" if "Ours" in name else "--",
                    color=PAL.get(name,"#888"),
                    label=f"{name}  (AUC={ra:.3f})")
        except Exception: pass
    ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Fig. 3 — ROC Curves  (NSL-KDD, macro one-vs-rest)",fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.2)
    _save(fig,"fig3_roc")

def fig4_shap(sv_nn, sv_rf, feat_names, rf_feat_names, top=15):
    n = min(top, len(feat_names), len(sv_nn))
    m = min(top, len(rf_feat_names), len(sv_rf))
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax,(sv,fn,title,nn) in zip(axes,[
        (sv_nn, feat_names,    "Neural Path — NeuroSym-ZTA", n),
        (sv_rf, rf_feat_names, "Random Forest Baseline",     m)]):
        idx = np.argsort(sv[:nn] if len(sv)>=nn else sv)[::-1][:nn]
        ax.barh(range(len(idx)), sv[idx[::-1]], color="#1f4e79", alpha=0.83, edgecolor="white")
        ax.set_yticks(range(len(idx))); ax.set_yticklabels([fn[i] for i in idx[::-1]],fontsize=8)
        ax.set_xlabel("Mean |Attribution|  (DENY class)")
        ax.set_title(f"SHAP — {title}",fontweight="bold",fontsize=10)
        ax.grid(axis="x",alpha=0.2)
    fig.suptitle("Fig. 4 — Feature Attribution for DENY Decision\n"
                 "(Left: Gradient×Input — neural path;  Right: TreeSHAP — RF baseline)",
                 fontweight="bold",fontsize=11,y=1.04)
    _save(fig,"fig4_shap")

def fig5_ablation(abl):
    configs = list(abl.keys())
    metrics = ["Accuracy","F1","AUC"]
    x = np.arange(len(configs)); w = 0.25
    colors = ["#1f4e79","#2e75b6","#70ad47"]
    fig, ax = plt.subplots(figsize=(10,4.8))
    for i,m in enumerate(metrics):
        vals = [abl[c].get(m,0) if not (isinstance(abl[c].get(m,0),float) and
                np.isnan(abl[c].get(m,0))) else 0 for c in configs]
        ax.bar(x+(i-1)*w, vals, w, label=m, color=colors[i], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim([0.40,1.02])
    ax.set_title("Fig. 5 — Ablation Study  (NSL-KDD Test Set)",fontweight="bold")
    ax.legend(); ax.grid(axis="y",alpha=0.2)
    _save(fig,"fig5_ablation")

def fig6_rule_weights(w):
    fig, ax = plt.subplots(figsize=(7,4))
    cols = ["#1f4e79" if v>=0 else "#c55a11" for v in w]
    ax.bar(range(N_RULES), w, color=cols, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(N_RULES))
    ax.set_xticklabels(["R1\nAuth-Fail","R2\nPriv-Esc","R3\nConn-Err",
                         "R4\nSvc-Div","R5\nCompromise"],fontsize=10)
    ax.axhline(0,color="black",lw=0.8)
    ax.set_ylabel("Learned Weight (wk)")
    ax.set_title("Fig. 6 — Learned Symbolic Rule Weights\n"
                 "Positive = risk-increasing  |  Negative = risk-reducing",fontweight="bold")
    ax.grid(axis="y",alpha=0.2)
    _save(fig,"fig6_rule_weights")

def fig7_training(hist):
    ep = range(1, len(hist["loss"])+1)
    fig,(a1,a2) = plt.subplots(1,2,figsize=(9,4))
    a1.plot(ep,hist["loss"],"#1f4e79",lw=2,label="Train Loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Cross-Entropy Loss")
    a1.set_title("Training Loss"); a1.grid(alpha=0.2); a1.legend()
    a2.plot(ep,hist["val_acc"],"#1f4e79",lw=2,label="Val Accuracy")
    a2.plot(ep,hist["val_f1"],"#ed7d31",lw=2,ls="--",label="Val F1 (macro)")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Score"); a2.set_ylim([0.3,1.02])
    a2.set_title("Validation Metrics"); a2.grid(alpha=0.2); a2.legend()
    fig.suptitle("Fig. 7 — NeuroSym-ZTA Training Convergence  (NSL-KDD)",
                 fontweight="bold",fontsize=11,y=1.03)
    _save(fig,"fig7_training")

# ═══════════════════════════════════════════════════════════════════════════
# 13.  RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════
def results_table(res):
    rows=[]
    for name,r in res.items():
        auc = r.get("AUC",float("nan"))
        rows.append({"Method":name,
                     "Accuracy": f"{r.get('Accuracy',0):.4f}",
                     "F1 (macro)":f"{r.get('F1',0):.4f}",
                     "Precision": f"{r.get('Precision',0):.4f}",
                     "Recall":    f"{r.get('Recall',0):.4f}",
                     "AUC-ROC":   f"{auc:.4f}" if not (isinstance(auc,float) and np.isnan(auc)) else "n/a"})
    df = pd.DataFrame(rows)
    print("\n"+"="*74)
    print("TABLE I  —  ACCESS DECISION PERFORMANCE  (NSL-KDD Test Set)")
    print("="*74)
    print(df.to_string(index=False))
    print("="*74)
    df.to_csv(f"{RESULTS_DIR}/results_table.csv", index=False)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 14.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    if not ensure_data(): sys.exit(1)

    X_tr,X_te,y_tr,y_te,sc,df_tr,df_te = load_preprocess()
    in_dim = X_tr.shape[1]

    print("\nComputing rule activations …")
    R_tr = compute_rules(df_tr)
    R_te = compute_rules(df_te)

    print("\nBuilding sequences (shuffle → centre label) …")
    Xs_tr,Rs_tr,ys_tr = make_sequences(X_tr, R_tr, y_tr)
    Xs_te,Rs_te,ys_te = make_sequences(X_te, R_te, y_te)
    print(f"  Train seqs: {len(Xs_tr):,}  |  Test seqs: {len(Xs_te):,}")
    bc = np.bincount(ys_te, minlength=3)
    print(f"  Test seq dist → ALLOW:{bc[0]}  CHALLENGE:{bc[1]}  DENY:{bc[2]}")

    # 80/20 train/val split
    rng  = np.random.RandomState(SEED)
    perm = rng.permutation(len(Xs_tr))
    nv   = int(len(Xs_tr)*0.20)
    vi,ti = perm[:nv], perm[nv:]
    Xv,Rv,yv = Xs_tr[vi],Rs_tr[vi],ys_tr[vi]
    Xt,Rt,yt = Xs_tr[ti],Rs_tr[ti],ys_tr[ti]

    res_all={};  preds_all={}

    # 1 — proposed NeuroSym-ZTA
    print("\n[1/5]  NeuroSym-ZTA (Proposed) …")
    t0  = time.time()
    ns  = NeuroSymZTA(in_dim).to(DEVICE)
    ns, hist = train_model(ns, Xt,Rt,yt, Xv,Rv,yv, name="NeuroSym-ZTA")
    tt  = time.time()-t0
    print(f"  Training: {tt:.1f}s")
    m_ns = eval_model(ns, Xs_te,Rs_te,ys_te)
    res_all["NeuroSym-ZTA (Ours)"]   = m_ns
    preds_all["NeuroSym-ZTA (Ours)"] = m_ns["preds"]

    # 2 — standalone BiGRU
    print("\n[2/5]  Standalone BiGRU …")
    gru = StandaloneGRU(in_dim).to(DEVICE)
    gru,_ = train_model(gru, Xt,Rt,yt, Xv,Rv,yv, name="BiGRU")
    m_gru = eval_model(gru, Xs_te,Rs_te,ys_te)
    res_all["Standalone BiGRU"]   = m_gru
    preds_all["Standalone BiGRU"] = m_gru["preds"]

    # 3 — no symbolic layer (ablation)
    print("\n[3/5]  No Symbolic Layer (ablation) …")
    ns2 = NeuroSymZTA(in_dim).to(DEVICE)
    ns2,_ = train_model(ns2, Xt,Rt,yt, Xv,Rv,yv, name="NoSym")
    # evaluate using neural path only
    ns2.eval()
    with torch.no_grad():
        Xte_t=torch.FloatTensor(Xs_te).to(DEVICE); Rte_t=torch.FloatTensor(Rs_te).to(DEVICE)
        _,np_,_,_ = ns2(Xte_t,Rte_t)
        ns2_probs = np_.cpu().numpy()
    ns2_preds = ns2_probs.argmax(1)
    yb_te = label_binarize(ys_te,classes=[0,1,2])
    try: ns2_auc=roc_auc_score(yb_te,ns2_probs,average="macro",multi_class="ovr") if len(np.unique(ys_te))==3 else float("nan")
    except: ns2_auc=float("nan")
    m_ns2=dict(Accuracy=accuracy_score(ys_te,ns2_preds),
               F1=f1_score(ys_te,ns2_preds,average="macro",zero_division=0),
               Precision=precision_score(ys_te,ns2_preds,average="macro",zero_division=0),
               Recall=recall_score(ys_te,ns2_preds,average="macro",zero_division=0),
               AUC=ns2_auc, preds=ns2_preds, probs=ns2_probs)
    res_all["No Symbolic Layer"]   = m_ns2
    preds_all["No Symbolic Layer"] = ns2_preds

    # 4 — static fusion α=0.5 (ablation)
    print("\n[4/5]  Static Fusion α=0.5 (ablation) …")
    sf = NeuroSymZTA(in_dim).to(DEVICE)
    sf,_ = train_model(sf, Xt,Rt,yt, Xv,Rv,yv, name="StaticFusion")
    m_sf  = eval_static_fusion(sf, Xs_te,Rs_te,ys_te)
    res_all["Static Fusion (α=0.5)"]   = m_sf
    preds_all["Static Fusion (α=0.5)"] = m_sf["preds"]

    # 5 — random forest
    print("\n[5/5]  Random Forest …")
    rf, m_rf, Xfl_tr, Xfl_te = eval_rf(Xt,Rt,yt, Xs_te,Rs_te,ys_te)
    res_all["Random Forest"]   = m_rf
    preds_all["Random Forest"] = m_rf["preds"]

    # rule engine
    m_re = eval_rule_engine(Rs_te, ys_te)
    res_all["Rule Engine Only"]   = m_re
    preds_all["Rule Engine Only"] = m_re["preds"]

    # tables
    results_table(res_all)
    priv_esc_table(ys_te, preds_all)

    # SHAP
    print("\nRunning attribution analysis …")
    sv_nn = shap_gradient_input(ns, Xs_te)
    rf_feat_names = ALL_FEAT + [f"rule_{i+1}" for i in range(N_RULES)]
    sv_rf = shap_tree(rf, Xfl_te)

    # latency
    print("\nLatency benchmark …")
    lat = latency_bench(ns, Xs_te, Rs_te)

    # figures
    print("\nGenerating figures …")
    fig1_arch()
    fig2_confusion(ys_te, m_ns["preds"])
    fig3_roc(res_all, ys_te)
    # align SHAP lengths
    n_feat = min(len(ALL_FEAT), len(sv_nn))
    n_rff  = min(len(rf_feat_names), len(sv_rf))
    fig4_shap(sv_nn[:n_feat], sv_rf[:n_rff],
              ALL_FEAT[:n_feat], rf_feat_names[:n_rff])
    fig5_ablation({
        "NeuroSym-ZTA\n(Ours)":       res_all["NeuroSym-ZTA (Ours)"],
        "Static Fusion\n(α=0.5)":     res_all["Static Fusion (α=0.5)"],
        "No Symbolic\nLayer":         res_all["No Symbolic Layer"],
        "Standalone\nBiGRU":          res_all["Standalone BiGRU"],
        "Random\nForest":             res_all["Random Forest"],
        "Rule Engine\nOnly":          res_all["Rule Engine Only"],
    })
    fig6_rule_weights(ns.sym.weights())
    fig7_training(hist)

    # JSON summary
    def _s(v): return None if isinstance(v,float) and np.isnan(v) else v
    summary=dict(
        dataset="NSL-KDD (KDDTrain+.txt, KDDTest+.txt)",
        n_train_seq=int(len(Xt)), n_test_seq=int(len(Xs_te)),
        seq_len=SEQ_LEN, stride=STRIDE, input_dim=in_dim,
        hidden_dim=HIDDEN, n_rules=N_RULES, epochs=EPOCHS,
        device=str(DEVICE), train_time_s=round(tt,1),
        learned_rule_weights=ns.sym.weights().tolist(),
        latency_ms=lat,
        results={nm:{k:_s(v) for k,v in r.items() if k not in("preds","probs")}
                 for nm,r in res_all.items()})
    with open(f"{RESULTS_DIR}/summary.json","w") as f:
        json.dump(summary,f,indent=2)

    ours = res_all["NeuroSym-ZTA (Ours)"]
    print("\n"+"="*65)
    print("Phase 2 COMPLETE")
    print(f"  Acc={ours['Accuracy']:.4f}  F1={ours['F1']:.4f}  AUC={_s(ours['AUC'])}")
    print(f"  Training time: {tt:.1f}s  |  Device: {DEVICE}")
    print("Files in ./results/:")
    for fn in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {fn:<38} {os.path.getsize(f'{RESULTS_DIR}/{fn}')//1024:>5} KB")
    print("="*65)

if __name__=="__main__":
    main()
