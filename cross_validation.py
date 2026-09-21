"""
5-Fold Stratified Cross-Validation on BraTS + Figshare combined dataset.
Saves all results to D:\BrainTumor Revision\results\

Reviewer requirement:
  "Using a single dataset is not enough. Perform experiments with additional
   datasets. Cross validation should be used."
"""
import os, sys, glob, random, csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r"D:\BrainTumor Revision")
from BrainTumorQANN import QCNN, simple_preprocess

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------- CONFIG ----------
FINETUNED_WEIGHTS = r"D:\Brain\Results_Finetune\finetuned_qcnn_best.pth"
PRETRAINED_WEIGHTS = r"D:\Brain\Results\best_qcnn.pth"
RESULTS_DIR = r"D:\BrainTumor Revision\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FIGSHARE_YES = r"D:\Brain\Datasets\Figshare\classification\yes"
FIGSHARE_NO  = r"D:\Brain\Datasets\Figshare\classification\no"
BRATS_YES    = r"D:\Brain\Datasets\BraTS\classification\yes"
BRATS_NO     = r"D:\Brain\Datasets\BraTS\classification\no"
TCGA_YES     = r"D:\Brain\Datasets\TCGA_LGG\classification\yes"
TCGA_NO      = r"D:\Brain\Datasets\TCGA_LGG\classification\no"

K_FOLDS    = 5
EPOCHS     = 15
BATCH_SIZE = 16
LR         = 5e-5
PATIENCE   = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- DATASET ----------
class CVDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files   = list(files)
        self.labels  = list(labels)
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            arr = np.array(Image.open(self.files[idx]).convert('L').resize((224,224), Image.BILINEAR), np.float32) / 255.0
            arr = cv2.equalizeHist((arr*255).astype(np.uint8)).astype(np.float32) / 255.0
            arr = simple_preprocess(arr)
            if self.augment:
                if random.random() > 0.5: arr = np.fliplr(arr)
                if random.random() > 0.5: arr = np.flipud(arr)
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
                arr = cv2.warpAffine(arr, M, (224, 224))
            return torch.from_numpy(arr.copy()).unsqueeze(0).float(), int(self.labels[idx])
        except Exception:
            return torch.zeros(1, 224, 224), int(self.labels[idx])


def load_all_files():
    sources = [
        (FIGSHARE_YES, 1, "Figshare-tumor"),
        (FIGSHARE_NO,  0, "Figshare-normal"),
        (BRATS_YES,    1, "BraTS-tumor"),
        (BRATS_NO,     0, "BraTS-normal"),
        (TCGA_YES,     1, "TCGA-LGG-tumor"),
        (TCGA_NO,      0, "TCGA-LGG-normal"),
    ]
    all_files, all_labels = [], []
    print("\nDataset inventory:")
    print("-" * 40)
    total = 0
    for folder, label, name in sources:
        files = (glob.glob(os.path.join(folder, '*.[pj][np]g')) +
                 glob.glob(os.path.join(folder, '*.tif'))) if os.path.exists(folder) else []
        all_files.extend(files)
        all_labels.extend([label] * len(files))
        print(f"  {name:<22}: {len(files):>5} images")
        total += len(files)
    print(f"  {'TOTAL':<22}: {total:>5} images")
    print(f"  Tumor(1): {sum(all_labels)}  |  Normal(0): {total - sum(all_labels)}")
    print("-" * 40)
    return all_files, all_labels


# ---------- TRAINING ----------
def train_fold(model, train_loader, val_loader, fold_num):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_acc, best_state, patience_ctr = 0.0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(imgs), labs).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                preds_all.extend(torch.argmax(model(imgs.to(DEVICE)), 1).cpu().numpy())
                labels_all.extend(labs.numpy())

        acc = accuracy_score(labels_all, preds_all)
        print(f"    Fold {fold_num} | Epoch {epoch+1:02d}/{EPOCHS} | Val Acc={acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_fold(model, val_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labs in val_loader:
            logits = model(imgs.to(DEVICE))
            all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(labs.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------- MAIN ----------
def run_cv(all_files, all_labels):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    all_files  = np.array(all_files)
    all_labels = np.array(all_labels)

    start_weights = FINETUNED_WEIGHTS if os.path.exists(FINETUNED_WEIGHTS) else PRETRAINED_WEIGHTS
    print(f"\nStarting weights: {start_weights}")

    fold_metrics = []
    all_cms = []

    print(f"\n{'='*60}")
    print(f"  STRATIFIED {K_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'='*60}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels), 1):
        print(f"\n--- Fold {fold_idx}/{K_FOLDS} ---")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

        train_ds = CVDataset(all_files[train_idx], all_labels[train_idx], augment=True)
        val_ds   = CVDataset(all_files[val_idx],   all_labels[val_idx],   augment=False)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)

        model = QCNN(10, 2).to(DEVICE)
        model.load_state_dict(torch.load(start_weights, map_location=DEVICE))
        model = train_fold(model, train_loader, val_loader, fold_idx)

        fold_path = os.path.join(RESULTS_DIR, f"qcnn_fold_{fold_idx}.pth")
        torch.save(model.state_dict(), fold_path)

        labels, preds, probs = evaluate_fold(model, val_loader)
        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1   = f1_score(labels, preds, zero_division=0)
        cm   = confusion_matrix(labels, preds)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            spec = 0.0

        try:    auc = roc_auc_score(labels, probs)
        except: auc = float('nan')

        metrics = dict(fold=fold_idx, acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, auc=auc)
        fold_metrics.append(metrics)
        all_cms.append(cm)

        print(f"\n  Fold {fold_idx} Results:")
        print(f"    Accuracy    : {acc*100:.2f}%")
        print(f"    Precision   : {prec:.4f}")
        print(f"    Recall/Sens : {rec:.4f}")
        print(f"    Specificity : {spec:.4f}")
        print(f"    F1 Score    : {f1:.4f}")
        print(f"    AUC-ROC     : {auc:.4f}")
        print(f"    Confusion Matrix:\n{cm}")

    return fold_metrics, all_cms


def save_summary(fold_metrics, all_cms):
    keys = ['acc', 'prec', 'rec', 'spec', 'f1', 'auc']
    names = {'acc': 'Accuracy', 'prec': 'Precision', 'rec': 'Recall/Sensitivity',
             'spec': 'Specificity', 'f1': 'F1 Score', 'auc': 'AUC-ROC'}

    print(f"\n{'='*65}")
    print(f"  {K_FOLDS}-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
    print(f"  {'-'*56}")

    csv_rows = []
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        mean, std = np.mean(vals), np.std(vals)
        mn, mx = np.min(vals), np.max(vals)
        scale = 100 if k == 'acc' else 1
        unit = '%' if k == 'acc' else ''
        print(f"  {names[k]:<22} {mean*scale:>8.2f}{unit}  {std*scale:>9.4f}  "
              f"{mn*scale:>8.2f}{unit}  {mx*scale:>8.2f}{unit}")
        csv_rows.append({'Metric': names[k], 'Mean': f"{mean*scale:.4f}",
                         'Std': f"{std*scale:.4f}", 'Min': f"{mn*scale:.4f}", 'Max': f"{mx*scale:.4f}"})

    print(f"\n  Per-Fold Breakdown:")
    print(f"  {'Fold':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*54}")
    for m in fold_metrics:
        print(f"  {m['fold']:<6} {m['acc']*100:>7.2f}% {m['prec']:>8.4f} {m['rec']:>8.4f} "
              f"{m['spec']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")
    print(f"{'='*65}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "cv_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'Mean', 'Std', 'Min', 'Max'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved: {csv_path}")

    # Per-fold results CSV
    fold_csv = os.path.join(RESULTS_DIR, "cv_per_fold.csv")
    with open(fold_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'acc', 'prec', 'rec', 'spec', 'f1', 'auc'])
        writer.writeheader()
        writer.writerows(fold_metrics)
    print(f"Per-fold CSV saved: {fold_csv}")

    # Bar chart per fold
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'QCNN {K_FOLDS}-Fold Cross-Validation (BraTS + Figshare)', fontsize=14, fontweight='bold')
    for ax, k in zip(axes.flat, keys):
        vals = [m[k] for m in fold_metrics]
        scale = 100 if k == 'acc' else 1
        bars = ax.bar([f'Fold {m["fold"]}' for m in fold_metrics],
                      [v*scale for v in vals], color='steelblue', edgecolor='black')
        mean = np.mean(vals)*scale
        ax.axhline(mean, color='red', linestyle='--', label=f'Mean={mean:.2f}{"%" if k=="acc" else ""}')
        ax.set_title(names[k], fontweight='bold')
        ax.set_ylabel('%' if k == 'acc' else 'Score')
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{v*scale:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, 'cv_metrics_per_fold.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Bar chart saved: {bar_path}")

    # Average confusion matrix
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    plt.figure(figsize=(6, 5))
    sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'], annot_kws={'size': 14})
    plt.title(f'Average Confusion Matrix ({K_FOLDS}-Fold CV)', fontsize=12, fontweight='bold')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'cv_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    print(f"\nAll CV results saved to: {RESULTS_DIR}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Results -> {RESULTS_DIR}")
    all_files, all_labels = load_all_files()
    if len(all_files) == 0:
        print("ERROR: No images found."); return
    fold_metrics, all_cms = run_cv(all_files, all_labels)
    save_summary(fold_metrics, all_cms)
    print("\nDone! Use cv_results.csv and plots in your paper revision.")


if __name__ == "__main__":
    main()
