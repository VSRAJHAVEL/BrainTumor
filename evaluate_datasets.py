import os, sys, glob, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv

sys.path.insert(0, r"D:\BrainTumor Revision")
from BrainTumorQANN import QCNN, simple_preprocess

WEIGHTS     = r"D:\Brain\Results_Finetune\finetuned_qcnn_best.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = r"D:\BrainTumor Revision\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "Figshare": {
        "yes": r"D:\Brain\Datasets\Figshare\classification\yes",
        "no":  r"D:\Brain\Datasets\Figshare\classification\no",
    },
    "BraTS 2021": {
        "yes": r"D:\Brain\Datasets\BraTS\classification\yes",
        "no":  r"D:\Brain\Datasets\BraTS\classification\no",
    },
    "TCGA-LGG (NIH)": {
        "yes": r"D:\Brain\Datasets\TCGA_LGG\classification\yes",
        "no":  r"D:\Brain\Datasets\TCGA_LGG\classification\no",
    },
}


class SimpleDS(Dataset):
    def __init__(self, yes_dir, no_dir=None):
        yf = glob.glob(os.path.join(yes_dir, '*.[pj][np]g')) + glob.glob(os.path.join(yes_dir, '*.tif')) if os.path.exists(yes_dir) else []
        nf = glob.glob(os.path.join(no_dir,  '*.[pj][np]g')) + glob.glob(os.path.join(no_dir,  '*.tif')) if (no_dir and os.path.exists(no_dir)) else []
        self.files  = yf + nf
        self.labels = [1]*len(yf) + [0]*len(nf)
        print("  Tumor:", len(yf), " Normal:", len(nf))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            arr = np.array(Image.open(self.files[idx]).convert('L').resize((224,224), Image.BILINEAR), np.float32) / 255.0
            arr = cv2.equalizeHist((arr*255).astype(np.uint8)).astype(np.float32) / 255.0
            arr = simple_preprocess(arr)
            return torch.from_numpy(arr.copy()).unsqueeze(0).float(), int(self.labels[idx])
        except Exception:
            return torch.zeros(1, 224, 224), int(self.labels[idx])


def save_confusion_matrix(cm, name, labels=('No Tumor', 'Tumor')):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title('Confusion Matrix - ' + name, fontsize=12, fontweight='bold')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'cm_' + name.replace(' ', '_') + '.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print("  Saved:", fname)


def evaluate(model, name, yes_dir, no_dir=None):
    print("\n" + "="*50)
    print("Dataset:", name)
    print("="*50)
    ds = SimpleDS(yes_dir, no_dir)
    if len(ds) == 0:
        print("No images found."); return None
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    preds, labels, probs = [], [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            try:
                logits = model(x.to(DEVICE))
                prob   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred   = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                labels.extend(y.tolist())
                probs.extend(prob.tolist())
            except Exception as e:
                print("  Batch", i, "error:", str(e)[:60])
            if (i+1) % 100 == 0:
                print("  Processed", (i+1)*4, "/", len(ds), "images...")
    if len(preds) == 0:
        print("No predictions made."); return None

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)
    try:    auc = roc_auc_score(labels, probs)
    except: auc = float('nan')
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn/(tn+fp) if (tn+fp) > 0 else float('nan')
    else:
        spec = float('nan')

    print("Accuracy   :", round(acc*100, 2), "%")
    print("Precision  :", round(prec, 4))
    print("Recall     :", round(rec, 4))
    print("Specificity:", round(spec, 4))
    print("F1 Score   :", round(f1, 4))
    print("AUC-ROC    :", round(auc, 4))
    print("Confusion Matrix:\n", cm)
    unique = sorted(set(labels))
    tnames = ['No Tumor', 'Tumor'] if len(unique) == 2 else (['No Tumor'] if unique == [0] else ['Tumor'])
    print(classification_report(labels, preds, labels=unique, target_names=tnames, digits=4))

    save_confusion_matrix(cm, name)
    return dict(name=name, acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, auc=auc,
                n_total=len(ds), n_tumor=sum(labels), n_normal=len(labels)-sum(labels))


def main():
    print("Device:", DEVICE)
    print("Weights:", WEIGHTS)
    model = QCNN(10, 2).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    print("Model loaded OK\n")

    results = []
    for name, paths in DATASETS.items():
        r = evaluate(model, name, paths["yes"], paths.get("no"))
        if r:
            results.append(r)

    if not results:
        print("No results collected."); return

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, "dataset_results.csv")
    fieldnames = ['name', 'n_total', 'n_tumor', 'n_normal',
                  'acc', 'prec', 'rec', 'spec', 'f1', 'auc']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("\nCSV saved:", csv_path)

    # --- Bar chart comparison ---
    metrics = ['acc', 'prec', 'rec', 'f1', 'auc']
    labels_m = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        ax.bar(x + i*width, vals, width, label=r['name'])
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(labels_m)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('QCNN Performance on Additional Datasets (Fine-tuned)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "dataset_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print("Chart saved:", chart_path)

    # --- Summary table ---
    print("\n" + "="*75)
    print("SUMMARY TABLE - Fine-tuned QCNN on Additional Datasets")
    print("="*75)
    print(f"{'Dataset':<15} {'Images':>7} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
    print("-"*75)
    for r in results:
        print(f"{r['name']:<15} {r['n_total']:>7} {r['acc']*100:>7.2f}% "
              f"{r['prec']:>8.4f} {r['rec']:>8.4f} {r['spec']:>8.4f} "
              f"{r['f1']:>8.4f} {r['auc']:>8.4f}")
    print("="*75)
    print("\nAll results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
