# -*- coding: utf-8 -*-
"""
Fine-tune the pretrained QCNN on BraTS + Figshare datasets.
Strategy:
  - Load pretrained Br35H weights
  - Freeze the early conv layers (they already learned good features)
  - Fine-tune the upper layers + quantum + classifier on the new data
  - Use a low learning rate to avoid catastrophic forgetting
"""
import os
import sys
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\BrainTumor Revision")
from BrainTumorQANN import QCNN, simple_preprocess

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ----------------------- CONFIG -----------------------
PRETRAINED_PATH = r"D:\Brain\Results\best_qcnn.pth"
RESULTS_DIR     = r"D:\Brain\Results_Finetune"
os.makedirs(RESULTS_DIR, exist_ok=True)

FIGSHARE_YES = r"D:\Brain\Datasets\Figshare\classification\yes"
FIGSHARE_NO  = r"D:\Brain\Datasets\Figshare\classification\no"
BRATS_YES    = r"D:\Brain\Datasets\BraTS\classification\yes"
BRATS_NO     = r"D:\Brain\Datasets\BraTS\classification\no"

# Hyperparameters for fine-tuning (low LR to avoid forgetting)
BATCH_SIZE  = 16
EPOCHS      = 30
LR_FROZEN   = 1e-4   # learning rate when conv layers are frozen
LR_FULL     = 3e-5   # learning rate for full unfrozen fine-tuning
VAL_SPLIT   = 0.2
PATIENCE    = 10

# ----------------------- DATASET -----------------------
class FinetuneDataset(Dataset):
    def __init__(self, yes_folder, no_folder=None, size=(224, 224), augment=False):
        yes_files = glob.glob(os.path.join(yes_folder, '*.[pj][np]g')) if os.path.exists(yes_folder) else []
        no_files  = glob.glob(os.path.join(no_folder,  '*.[pj][np]g')) if (no_folder and os.path.exists(no_folder)) else []
        self.files  = yes_files + no_files
        self.labels = [1]*len(yes_files) + [0]*len(no_files)
        self.size   = size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L').resize(self.size, Image.BILINEAR)
        arr = np.array(img, np.float32) / 255.0

        # Histogram-equalize to bring Figshare/BraTS closer to Br35H distribution
        arr_uint = (arr * 255).astype(np.uint8)
        arr_eq   = cv2.equalizeHist(arr_uint).astype(np.float32) / 255.0

        # Standard QCNN preprocessing (CLAHE + normalize to [-1,1])
        arr_eq = simple_preprocess(arr_eq)

        # Augmentation during training
        if self.augment:
            if random.random() > 0.5:
                arr_eq = np.fliplr(arr_eq)
            if random.random() > 0.5:
                arr_eq = np.flipud(arr_eq)
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((arr_eq.shape[1]//2, arr_eq.shape[0]//2), angle, 1.0)
            arr_eq = cv2.warpAffine(arr_eq, M, (arr_eq.shape[1], arr_eq.shape[0]))

        tensor = torch.from_numpy(arr_eq.copy()).unsqueeze(0).float()
        return tensor, int(self.labels[idx])


# ----------------------- TRAINING LOOP -----------------------
def run_epoch_train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labs in tqdm(loader, desc="  Train", leave=False):
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = F.cross_entropy(out, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def run_epoch_val(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labs.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1, all_labels, all_preds


# ----------------------- MAIN -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Build combined dataset --
    print("\nLoading datasets...")
    figshare_ds = FinetuneDataset(FIGSHARE_YES, FIGSHARE_NO, augment=False)
    brats_ds    = FinetuneDataset(BRATS_YES,    BRATS_NO,    augment=False)
    combined    = ConcatDataset([figshare_ds, brats_ds])
    print(f"  Figshare: {len(figshare_ds)} images")
    print(f"  BraTS   : {len(brats_ds)} images")
    print(f"  Total   : {len(combined)} images")

    # Train / val split
    val_size   = int(len(combined) * VAL_SPLIT)
    train_size = len(combined) - val_size
    train_ds, val_ds = random_split(combined, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\n  Train samples: {train_size} | Val samples: {val_size}")

    # -- Load pretrained model --
    print(f"\nLoading pretrained weights from {PRETRAINED_PATH} ...")
    model = QCNN(10, 2).to(device)
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    print("  Loaded successfully.")

    # -- Phase 1: Freeze conv layers, fine-tune upper layers only --
    print("\n--- Phase 1: Fine-tuning (conv1-3 frozen) ---")
    for name, param in model.named_parameters():
        if any(name.startswith(f"conv{i}") for i in [1, 2, 3]):
            param.requires_grad = False

    frozen_params  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params: {frozen_params:,} | Trainable params: {trainable_params:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_FROZEN, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, verbose=True)

    best_acc, best_f1, pat_counter = 0.0, 0.0, 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(EPOCHS):
        loss = run_epoch_train(model, train_loader, optimizer, device)
        acc, f1, _, _ = run_epoch_val(model, val_loader, device)
        scheduler.step(1 - acc)
        history['train_loss'].append(loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Loss={loss:.4f} | Val Acc={acc*100:.2f}% | F1={f1:.4f}")

        if acc > best_acc:
            best_acc, best_f1, pat_counter = acc, f1, 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "finetuned_qcnn_phase1.pth"))
            print("    *** Best saved! ***")
        else:
            pat_counter += 1
            if pat_counter >= PATIENCE:
                print("  Early stop."); break

    # -- Phase 2: Unfreeze everything, full fine-tune at very low LR --
    print("\n--- Phase 2: Full fine-tuning (all layers, very low LR) ---")
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "finetuned_qcnn_phase1.pth")))
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, pat_counter = 0.0, 0
    for epoch in range(EPOCHS):
        loss = run_epoch_train(model, train_loader, optimizer, device)
        acc, f1, all_labels, all_preds = run_epoch_val(model, val_loader, device)
        scheduler.step()
        history['train_loss'].append(loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Loss={loss:.4f} | Val Acc={acc*100:.2f}% | F1={f1:.4f}")

        if acc > best_acc:
            best_acc, pat_counter = acc, 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "finetuned_qcnn_best.pth"))
            print("    *** Best saved! ***")
        else:
            pat_counter += 1
            if pat_counter >= PATIENCE:
                print("  Early stop."); break

    # -- Final evaluation --
    print("\n" + "="*60)
    print("  FINAL FINE-TUNED MODEL EVALUATION")
    print("="*60)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "finetuned_qcnn_best.pth")))
    acc, f1, all_labels, all_preds = run_epoch_val(model, val_loader, device)
    print(f"  Val Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Val F1 Score : {f1:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['No Tumor','Tumor'], digits=4)}")

    # -- Plot training curves --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot([v*100 for v in history['val_acc']], label='Val Acc (%)')
    axes[1].plot([v*100 for v in history['val_f1']], label='Val F1 (%)')
    axes[1].set_title('Validation Metrics'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'finetune_curves.png'), dpi=150)
    plt.close()

    print(f"\nFine-tuned model saved to: {RESULTS_DIR}")
    print(f"Best model: {os.path.join(RESULTS_DIR, 'finetuned_qcnn_best.pth')}")
    print("="*60)


if __name__ == "__main__":
    main()
