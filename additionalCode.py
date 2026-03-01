import os, glob, random, math, time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
import pandas as pd
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
CLASS_TO_IDX = {"no": 0, "yes": 1}  # no tumor=0, tumor=1

def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def auto_roi_crop(img_bgr, pad=10):
    """
    Heuristic ROI: find largest foreground region (brain area) and crop.
    Not tumor-mask; just tighter brain crop.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure foreground is white
    if th.mean() > 127:
        th = 255 - th

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return img_bgr  # fallback

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

class Br35HDataset(Dataset):
    def __init__(self, root, use_roi=False, img_size=224):
        self.root = root
        self.use_roi = use_roi
        self.img_size = img_size

        self.samples = []
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(root, cls_name)
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    self.samples.append((p, cls_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found under: {root}. Check folder structure.")

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {path}")

        img_bgr = clahe_rgb(img_bgr)

        if self.use_roi:
            img_bgr = auto_roi_crop(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        x = self.tf(img_pil)
        return x, torch.tensor(y, dtype=torch.long)
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        avg = x.mean(dim=(2,3))
        mx  = x.amax(dim=(2,3))
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

class CBAMLite(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
class SmallCNN(nn.Module):
    def __init__(self, use_attention=False, feat_dim=128):
        super().__init__()
        self.use_attention = use_attention

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.attn = CBAMLite(128) if use_attention else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attn(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)   # [B, 128]
        feat = self.fc(x)                         # [B, feat_dim]
        return feat
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# ---------- Quantum Layer (manual batching + dtype safe) ----------
class QuantumLayer(nn.Module):
    """
    Manual batched quantum layer:
    - avoids TorchLayer batching errors
    - converts list outputs to tensors
    - ensures float32 dtype
    """
    def __init__(self, n_qubits=10, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # inputs: [n_qubits] for ONE sample
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                # ring entanglement
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # returns list of expectation values (one per qubit)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

        # ✅ float32 trainable weights
        self.weights = nn.Parameter((0.01 * torch.randn(n_layers, n_qubits, 2)).float())

    def forward(self, x):
        """
        x: [B, n_qubits] (float32 expected)
        return: [B, n_qubits]
        """
        # Ensure inputs are not treated as trainable by PennyLane
        x = x.detach().float()

        outs = []
        for i in range(x.shape[0]):
            out_i = self.circuit(x[i], self.weights)  # list or tensor
            if isinstance(out_i, (list, tuple)):
                out_i = torch.stack(out_i)            # [n_qubits]
            outs.append(out_i.float())

        return torch.stack(outs, dim=0).float()        # [B, n_qubits]


# ---------- Classifier (classical or hybrid quantum) ----------
class Classifier(nn.Module):
    """
    Uses your existing backbone: SmallCNN(use_attention=..., feat_dim=128)
    Then:
      - classical head OR
      - hybrid quantum fusion head
    """
    def __init__(self, backbone, use_quantum=False, n_qubits=10):
        super().__init__()
        self.use_quantum = use_quantum
        self.backbone = backbone  # must output [B,128]

        if use_quantum:
            self.to_q = nn.Linear(128, n_qubits)
            self.q = QuantumLayer(n_qubits=n_qubits, n_layers=2)
            self.head = nn.Sequential(
                nn.Linear(128 + n_qubits, 64), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

    def forward(self, x):
        feat = self.backbone(x).float()  # [B,128]

        if self.use_quantum:
            q_in = (torch.tanh(self.to_q(feat)) * math.pi).float()  # [B,n_qubits]
            q_out = self.q(q_in).float()                            # [B,n_qubits]
            fused = torch.cat([feat, q_out], dim=1).float()
            return self.head(fused)

        return self.head(feat)
import gc, torch
gc.collect()
torch.cuda.empty_cache()
class Classifier(nn.Module):
    def __init__(self, use_attention=False, use_quantum=False, n_qubits=10):
        super().__init__()
        self.use_quantum = use_quantum
        self.backbone = SmallCNN(use_attention=use_attention, feat_dim=128)

        if use_quantum:
            # map 128 -> n_qubits for quantum encoding
            self.to_q = nn.Linear(128, n_qubits)
            self.q = QuantumLayer(n_qubits=n_qubits, n_layers=2)
            # fuse classical feat (128) + quantum outputs (n_qubits)
            self.head = nn.Sequential(
                nn.Linear(128 + n_qubits, 64), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

    def forward(self, x):
        feat = self.backbone(x)  # [B,128]
        if self.use_quantum:
            q_in = torch.tanh(self.to_q(feat)) * math.pi  # bounded angles
            q_out = self.q(q_in)                          # [B,n_qubits]
            fused = torch.cat([feat, q_out], dim=1)
            return self.head(fused)
        return self.head(feat)
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.numpy()
        logits = model(x).detach().cpu()
        pr = F.softmax(logits, dim=1)[:,1].numpy()
        pred = logits.argmax(dim=1).numpy()

        ys.extend(y.tolist())
        ps.extend(pred.tolist())
        probs.extend(pr.tolist())

    y_true = np.array(ys)
    y_pred = np.array(ps)
    y_prob = np.array(probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc  = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")
    return acc, prec, rec, f1, auc

def train_one_fold(model, train_loader, val_loader, device, epochs=15, lr=2e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0
    best_state = None
    patience, bad = 5, 0

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        acc, prec, rec, f1, auc = evaluate(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
def run_5fold(
    data_root="/content/Br35H",
    use_roi=False,
    use_attention=False,
    use_quantum=False,
    batch_size=16,
    epochs=15,
    lr=2e-4,
    seed=42
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Br35HDataset(data_root, use_roi=use_roi, img_size=224)
    labels = np.array([y for _, y in ds.samples])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        tr = Subset(ds, tr_idx)
        te = Subset(ds, te_idx)

        train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = Classifier(use_attention=use_attention, use_quantum=use_quantum).to(device)
        model = train_one_fold(model, train_loader, test_loader, device, epochs=epochs, lr=lr)

        acc, prec, rec, f1, auc = evaluate(model, test_loader, device)
        fold_metrics.append([acc, prec, rec, f1, auc])
        print(f"Fold {fold}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    m = np.array(fold_metrics)
    mean = m.mean(axis=0)
    std  = m.std(axis=0)
    print("\n=== 5-Fold Summary (Mean ± Std) ===")
    print(f"Accuracy : {mean[0]*100:.2f} ± {std[0]*100:.2f}")
    print(f"Precision: {mean[1]*100:.2f} ± {std[1]*100:.2f}  (macro)")
    print(f"Recall   : {mean[2]*100:.2f} ± {std[2]*100:.2f}  (macro)")
    print(f"F1       : {mean[3]*100:.2f} ± {std[3]*100:.2f}  (macro)")
    print(f"AUC      : {mean[4]:.4f} ± {std[4]:.4f}")
    # also return a DataFrame with per-fold metrics for convenience
    df = pd.DataFrame(fold_metrics, columns=["accuracy", "precision", "recall", "f1", "auc"])
    return mean, std, df
# CNN only (no attention), full image, classical
run_5fold("/content/Br35H", use_roi=False, use_attention=False, use_quantum=False)

# CNN + Attention
run_5fold("/content/Br35H", use_roi=False, use_attention=True, use_quantum=False)
# Classical CNN (no quantum)
run_5fold("/content/Br35H", use_roi=False, use_attention=True, use_quantum=False)

# Hybrid QCNN (quantum + fusion)
run_5fold("/content/Br35H", use_roi=False, use_attention=True, use_quantum=True)
import math
import torch
import torch.nn as nn

class DAU_HQCNN(nn.Module):
    """
    Proposed model:
    Attention CNN (CBAM) + Hybrid QCNN (10 qubits) + fusion + classifier head
    """
    def __init__(self, use_attention=True, use_quantum=True, n_qubits=10):
        super().__init__()
        self.use_quantum = use_quantum

        # 1) CNN feature extractor with optional attention
        self.backbone = SmallCNN(use_attention=use_attention, feat_dim=128)

        # 2) Hybrid QCNN module (optional)
        if use_quantum:
            self.to_q = nn.Linear(128, n_qubits)         # classical -> quantum input
            self.q = QuantumLayer(n_qubits=n_qubits, n_layers=8)

            # fusion + classifier head
            self.head = nn.Sequential(
                nn.Linear(128 + n_qubits, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
        else:
            # classical-only head
            self.head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

    def forward(self, x):
        feat = self.backbone(x).float()  # [B,128]

        if self.use_quantum:
            # map classical features -> quantum angles
            q_in = (torch.tanh(self.to_q(feat)) * math.pi).float()  # [B,n_qubits]
            q_out = self.q(q_in).float()                            # [B,n_qubits]

            fused = torch.cat([feat, q_out], dim=1).float()         # [B,128+n_qubits]
            return self.head(fused)

        return self.head(feat)
model = DAU_HQCNN(
    use_attention=True,
    use_quantum=True
).to(device)
run_5fold(
    "/content/Br35H",
    use_attention=True,
    use_quantum=True,
    batch_size=8,
    epochs=8
)
# Hybrid QCNN (quantum + fusion)
mean, std, df = run_5fold(
    "/content/Br35H",
    use_attention=True,   # CBAM attention ON
    use_quantum=True,     # Hybrid QCNN ON
    batch_size=8,
    epochs=8
)

print("Mean:", mean)
print("Std :", std)
display(df.head())
# Full-image
run_5fold("/content/Br35H", use_roi=False, use_attention=True, use_quantum=True)

# ROI-based (auto ROI crop heuristic)
run_5fold("/content/Br35H", use_roi=True, use_attention=True, use_quantum=True)
