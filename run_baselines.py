# -*- coding: utf-8 -*-
"""
Entrenamiento de baselines (EEGNet + CNN-only) con la MISMA metodologia
que el modelo propuesto CNN-Transformer.

Mismos folds, mismos canales, mismo preprocesamiento, misma evaluacion.

Uso:
    .venv\Scripts\python.exe run_baselines.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import re, json, random, time, csv, copy
from pathlib import Path
from glob import glob
import numpy as np
import mne
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from baselines_models import EEGNet, CNNOnly

# ── Paths ──────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent
DATA_RAW = PROJ / "data" / "raw"
FOLDS_JSON = PROJ / "models" / "00_folds" / "Kfold5.json"
OUT_DIR = PROJ / "models" / "06_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config (identica al modelo propuesto) ──────────────────────────────
RANDOM_STATE = 42
EPOCHS = 60
BATCH_SIZE = 64
BASE_LR = 5e-4
WARMUP_EPOCHS = 4
PATIENCE = 8
VAL_SUBJECT_FRAC = 0.18
DO_NOTCH = True
DO_BANDPASS = False
BP_LO, BP_HI = 4.0, 38.0
DO_CAR = False
RESAMPLE_HZ = None
ZSCORE_PER_EPOCH = False
TMIN, TMAX = -1.0, 5.0
USE_WEIGHTED_SAMPLER = True
EXCLUDE_SUBJECTS = {38, 88, 89, 92, 100, 104}
EXPECTED_8 = ["C3", "C4", "Cz", "CP3", "CP4", "FC3", "FC4", "FCz"]
IMAGERY_RUNS_LR = {4, 8, 12}
SW_MODE = "tta"
TTA_SHIFTS_S = [-0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reproducibilidad ───────────────────────────────────────────────────
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def seed_worker(worker_id):
    np.random.seed(RANDOM_STATE + worker_id)
    random.seed(RANDOM_STATE + worker_id)


# ── Carga de datos (identica al notebook) ──────────────────────────────
def normalize_ch_name(name):
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()

NORMALIZED_TARGETS = [normalize_ch_name(c) for c in EXPECTED_8]

def pick_8_channels(raw):
    chs = raw.info["ch_names"]
    norm_map = {normalize_ch_name(ch): ch for ch in chs}
    picked = []
    for tn, to in zip(NORMALIZED_TARGETS, EXPECTED_8):
        if tn in norm_map:
            picked.append(norm_map[tn])
        else:
            raise RuntimeError(f"Canal '{to}' no encontrado")
    return raw.pick(picks=picked)

def subject_id_to_int(s):
    m = re.match(r"[Ss](\d+)", s)
    return int(m.group(1)) if m else -1

def load_subject_epochs(subject_id):
    subj_dir = DATA_RAW / subject_id
    edfs = []
    for r in [4, 8, 12]:
        edfs.extend(glob(str(subj_dir / f"{subject_id}R{r:02d}.edf")))
    edfs = sorted(edfs)
    if not edfs:
        return np.empty((0, 8, 1), dtype=np.float32), np.empty((0,), dtype=int), None

    X_list, y_list, sfreq_out = [], [], None
    for edf_path in edfs:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw = pick_8_channels(raw)
        if DO_NOTCH:
            raw.notch_filter(freqs=[60.0], picks="all", verbose="ERROR")
        if DO_BANDPASS:
            raw.filter(l_freq=BP_LO, h_freq=BP_HI, picks="all", verbose="ERROR")
        sfreq = raw.info["sfreq"]
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        keep = {k: v for k, v in event_id.items() if k in {"T1", "T2"}}
        if not keep:
            continue
        epochs = mne.Epochs(raw, events=events, event_id=keep,
                           tmin=TMIN, tmax=TMAX, baseline=None,
                           preload=True, verbose="ERROR")
        X = epochs.get_data().astype(np.float32)
        ev_codes = epochs.events[:, 2]
        inv = {v: k for k, v in keep.items()}
        y_run = np.array([0 if inv[c] == "T1" else 1 for c in ev_codes], dtype=int)
        X_list.append(X)
        y_list.append(y_run)
        sfreq_out = sfreq

    if not X_list:
        return np.empty((0, 8, 1), dtype=np.float32), np.empty((0,), dtype=int), None
    return np.concatenate(X_list), np.concatenate(y_list), sfreq_out

def standardize_all(X_tr, X_val, X_te):
    """Estandariza los 3 conjuntos usando stats de X_tr (antes de modificarlo)."""
    C = X_tr.shape[1]
    X_tr = X_tr.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_te = X_te.astype(np.float32)
    for c in range(C):
        mu = X_tr[:, c, :].mean()
        sd = max(X_tr[:, c, :].std(), 1e-6)
        X_tr[:, c, :] = (X_tr[:, c, :] - mu) / sd
        X_val[:, c, :] = (X_val[:, c, :] - mu) / sd
        X_te[:, c, :] = (X_te[:, c, :] - mu) / sd
    return X_tr, X_val, X_te

def load_fold_subjects(fold):
    with open(FOLDS_JSON, "r") as f:
        data = json.load(f)
    for item in data.get("folds", []):
        if int(item["fold"]) == fold:
            return list(item["train"]), list(item["test"])
    raise ValueError(f"Fold {fold} not found")

def build_subject_label_map(subject_ids):
    y_list = []
    for sid in subject_ids:
        _, ys, _ = load_subject_epochs(sid)
        if len(ys) == 0:
            y_list.append(0)
        else:
            y_list.append(int(np.argmax(np.bincount(ys, minlength=2))))
    return np.array(y_list, dtype=int)


# ── Loss y augment (identicas) ─────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=1.5):
        super().__init__()
        self.alpha = alpha / alpha.sum()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = nn.functional.log_softmax(logits, dim=-1)
        p = logp.exp()
        idx = torch.arange(target.shape[0], device=logits.device)
        pt = p[idx, target]
        logpt = logp[idx, target]
        at = self.alpha[target]
        return (- at * ((1 - pt) ** self.gamma) * logpt).mean()

def augment_batch(xb):
    B, C, T = xb.shape
    if np.random.rand() < 0.35:
        ms = max(1, int(T * 0.03))
        shifts = torch.randint(-ms, ms + 1, (B,), device=xb.device)
        for i in range(B):
            xb[i] = torch.roll(xb[i], int(shifts[i].item()), dims=-1)
    if np.random.rand() < 0.35:
        xb = xb + 0.03 * torch.randn_like(xb)
    if np.random.rand() < 0.15:
        for i in range(B):
            idx = torch.randperm(C, device=xb.device)[:1]
            xb[i, idx, :] = 0.0
    return xb

def tta_logits(model, X, sfreq, device):
    model.eval()
    T = X.shape[-1]
    out = []
    with torch.no_grad():
        for i in range(X.shape[0]):
            x0 = X[i]
            acc = []
            for sh in TTA_SHIFTS_S:
                shift = int(round(sh * sfreq))
                if shift == 0:
                    x = x0
                elif shift > 0:
                    x = np.pad(x0[:, shift:], ((0, 0), (0, shift)), mode="edge")[:, :T]
                else:
                    s = -shift
                    x = np.pad(x0[:, :-s], ((0, 0), (s, 0)), mode="edge")[:, :T]
                xb = torch.tensor(x[None], dtype=torch.float32, device=device)
                logit = model(xb).detach().cpu().numpy()[0]
                acc.append(logit)
            out.append(np.mean(np.stack(acc), axis=0))
    return np.stack(out)


# ── Entrenamiento de un fold (identico al propuesto) ───────────────────
def train_one_fold(fold, model_factory, model_name, device):
    seed_everything(RANDOM_STATE)

    train_sub, test_sub = load_fold_subjects(fold)
    train_sub = [s for s in train_sub if subject_id_to_int(s) not in EXCLUDE_SUBJECTS]
    test_sub = [s for s in test_sub if subject_id_to_int(s) not in EXCLUDE_SUBJECTS]

    tr_subjects = sorted(train_sub)
    y_dom = build_subject_label_map(tr_subjects)
    n_val = max(1, int(round(len(tr_subjects) * VAL_SUBJECT_FRAC)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val,
                                 random_state=RANDOM_STATE + fold)
    idx = np.arange(len(tr_subjects))
    _, val_idx = next(sss.split(idx, y_dom))
    val_subjects = sorted([tr_subjects[i] for i in val_idx])
    train_subjects = [s for s in tr_subjects if s not in val_subjects]

    # Cargar datos
    def load_set(sids, desc):
        Xl, yl = [], []
        sfreq = None
        for sid in tqdm(sids, desc=desc):
            Xs, ys, sf = load_subject_epochs(sid)
            if len(ys) == 0:
                continue
            Xl.append(Xs); yl.append(ys)
            sfreq = sf if sfreq is None else sfreq
        return np.concatenate(Xl), np.concatenate(yl), sfreq

    X_tr, y_tr, sfreq = load_set(train_subjects, f"[{model_name}] Train fold{fold}")
    X_val, y_val, _ = load_set(val_subjects, f"[{model_name}] Val fold{fold}")
    X_te, y_te, _ = load_set(test_sub, f"[{model_name}] Test fold{fold}")

    X_tr, X_val, X_te = standardize_all(X_tr, X_val, X_te)

    print(f"[{model_name} Fold {fold}] train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).long())
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).long())

    if USE_WEIGHTED_SAMPLER:
        yb_np = y_tr
        counts = np.bincount(yb_np, minlength=2).astype(float)
        w = np.array([1.0 / counts[y] for y in yb_np])
        w = w / w.mean()
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        len(yb_np), replacement=True)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          worker_init_fn=seed_worker)
    else:
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                          worker_init_fn=seed_worker)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                       worker_init_fn=seed_worker)

    # Modelo
    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)

    cc = np.bincount(y_tr, minlength=2).astype(np.float32)
    inv_w = cc.sum() / (2.0 * np.maximum(cc, 1.0))
    alpha = torch.tensor(inv_w, dtype=torch.float32, device=device)
    crit = FocalLoss(alpha=alpha, gamma=1.5)

    # LR scheduler
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        prog = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.1 + 0.5 * 0.9 * (1.0 + np.cos(np.pi * min(1, prog)))
    scheduler = LambdaLR(opt, lr_lambda)

    # Train loop
    best_f1, best_state, wait = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss, n_seen = 0.0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment_batch(xb)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(yb)
            n_seen += len(yb)
        tr_loss /= max(1, n_seen)

        # Val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = xb.to(device)
                p = model(xb).argmax(1).cpu().numpy()
                preds.append(p); gts.append(yb.numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        f1m = f1_score(gts, preds, average="macro")

        if ep % 10 == 0 or ep == 1:
            vacc = accuracy_score(gts, preds)
            print(f"  Ep {ep:3d} | loss={tr_loss:.4f} | val_acc={vacc:.4f} | val_f1={f1m:.4f}")

        if f1m > best_f1 + 1e-4:
            best_f1 = f1m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        scheduler.step()
        if wait >= PATIENCE:
            print(f"  Early stop ep {ep} (best val_f1={best_f1:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test - evaluacion directa (sin TTA, para comparacion justa)
    model.eval()
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te).long())
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds, gts_te = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            xb = xb.to(device)
            p = model(xb).argmax(1).cpu().numpy()
            preds.append(p); gts_te.append(yb.numpy())
    preds = np.concatenate(preds); gts_te = np.concatenate(gts_te)
    acc = accuracy_score(gts_te, preds)
    f1m = f1_score(gts_te, preds, average="macro")

    # Params y FLOPs
    params = sum(p.numel() for p in model.parameters())
    # FLOPs estimado con dummy forward
    try:
        from torch.utils.flop_counter import FlopCounterMode
        dummy = torch.randn(1, 8, X_te.shape[-1], device=device)
        with FlopCounterMode(model, display=False) as fc:
            model(dummy)
        flops = fc.get_total_flops()
    except Exception:
        flops = 0

    print(f"[{model_name} Fold {fold}] ACC={acc:.4f} | F1={f1m:.4f} | Params={params:,} | FLOPs={flops:,}")
    return acc, f1m, params, flops


# ── Main ───────────────────────────────────────────────────────────────
def run_5fold(model_name, model_factory):
    print(f"\n{'='*60}")
    print(f"  BASELINE: {model_name}  (5-fold subject-wise CV)")
    print(f"{'='*60}")

    accs, f1s, params, flops = [], [], 0, 0
    for fold in range(1, 6):
        a, f, p, fl = train_one_fold(fold, model_factory, model_name, DEVICE)
        accs.append(a); f1s.append(f)
        params = p; flops = fl

    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    std_acc = np.std(accs)
    std_f1 = np.std(f1s)

    print(f"\n--- {model_name} RESULTADOS 5-FOLD ---")
    print(f"ACC por fold:  {[f'{a:.4f}' for a in accs]}")
    print(f"F1  por fold:  {[f'{f:.4f}' for f in f1s]}")
    print(f"ACC: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"F1:  {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"Params: {params:,}  ({params/1e6:.3f}M)")
    print(f"GFLOPs: {flops/1e9:.4f}")

    # Guardar CSV
    csv_path = OUT_DIR / f"results_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "acc", "f1_macro", "params", "flops"])
        for i, (a, fm) in enumerate(zip(accs, f1s), 1):
            w.writerow([model_name, i, f"{a:.6f}", f"{fm:.6f}", params, flops])
        w.writerow([model_name, "mean", f"{mean_acc:.6f}", f"{mean_f1:.6f}", params, flops])
        w.writerow([model_name, "std", f"{std_acc:.6f}", f"{std_f1:.6f}", "", ""])
    print(f"Resultados guardados: {csv_path}")

    return {
        "model": model_name, "accs": accs, "f1s": f1s,
        "mean_acc": mean_acc, "mean_f1": mean_f1,
        "params": params, "flops": flops,
    }


if __name__ == "__main__":
    seed_everything(RANDOM_STATE)
    print(f"Device: {DEVICE}")

    results = []

    # 1. EEGNet
    r1 = run_5fold("EEGNet", lambda: EEGNet(n_ch=8, n_cls=2, sfreq=160, T=961,
                                              F1=16, D=4, F2=32, p_drop=0.25))
    results.append(r1)

    # 2. CNN-Only
    r2 = run_5fold("CNNOnly", lambda: CNNOnly(n_ch=8, n_cls=2))
    results.append(r2)

    # Tabla resumen final
    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA FINAL")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'ACC':>8} {'F1':>8} {'Params':>10} {'GFLOPs':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<25} {r['mean_acc']:.4f}   {r['mean_f1']:.4f}   "
              f"{r['params']/1e6:.3f}M     {r['flops']/1e9:.4f}")
    # Agregar modelo propuesto para referencia
    print(f"{'Proposed CNN-Transformer':<25} {'0.8226':>8} {'0.8221':>8} {'0.232M':>10} {'0.0243':>8}")
    print(f"{'='*70}")
