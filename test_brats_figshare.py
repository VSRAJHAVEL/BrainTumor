import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys

# Add the project directory to path to import QCNN
sys.path.insert(0, r"D:\BrainTumor Revision")

try:
    from BrainTumorQANN import QCNN, simple_preprocess
except ImportError as e:
    print(f"Could not import QCNN: {e}")
    exit(1)


class TestDataset(Dataset):
    def __init__(self, yes_folder, no_folder=None, size=(224, 224)):
        yes_files = glob.glob(os.path.join(yes_folder, '*.[pj][np]g')) if os.path.exists(yes_folder) else []
        no_files = glob.glob(os.path.join(no_folder, '*.[pj][np]g')) if (no_folder and os.path.exists(no_folder)) else []

        self.files = yes_files + no_files
        self.labels = [1] * len(yes_files) + [0] * len(no_files)
        self.size = size

        if not self.files:
            raise FileNotFoundError(f"No images found in {yes_folder} or {no_folder}")

        print(f"  Tumor (yes) images : {len(yes_files)}")
        print(f"  Normal (no) images : {len(no_files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = int(self.labels[idx])

        img_gray = Image.open(path).convert('L').resize(self.size, Image.BILINEAR)
        arr_gray = np.array(img_gray, np.float32) / 255.0
        arr_gray = simple_preprocess(arr_gray)
        tensor = torch.from_numpy(arr_gray).unsqueeze(0).float()
        return tensor, label


def test_model_on_dataset(model, dataloader, device, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\nRunning inference on {dataset_name} ...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx+1)*dataloader.batch_size} / {len(dataloader.dataset)} images...")

    acc   = accuracy_score(all_labels, all_preds)
    prec  = precision_score(all_labels, all_preds, zero_division=0)
    rec   = recall_score(all_labels, all_preds, zero_division=0)
    f1    = f1_score(all_labels, all_preds, zero_division=0)
    cm    = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*55}")
    print(f"  Results for: {dataset_name}")
    print(f"{'='*55}")
    print(f"  Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['No Tumor','Tumor'], digits=4)}")
    return acc, prec, rec, f1, cm


def main():
    # CNN backbone runs on GPU. The QuantumLayer internally converts its
    # tiny 10-value input to numpy, runs PennyLane on CPU, then moves
    # the result back to GPU. No CUDA/CPU conflict.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (CNN on GPU, quantum circuit bridged via numpy)")

    model_path = r"D:\Brain\Results\best_qcnn.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Pretrained model not found at {model_path}. Train the model first.")
        return

    model = QCNN(10, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded pretrained model from {model_path}\n")

    # ---------- FIGSHARE ----------
    figshare_yes = r"D:\Brain\Datasets\Figshare\classification\yes"
    figshare_no  = r"D:\Brain\Datasets\Figshare\classification\no"

    if os.path.exists(figshare_yes):
        try:
            print("Loading Figshare dataset...")
            ds = TestDataset(figshare_yes, figshare_no)
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            test_model_on_dataset(model, loader, device, "Figshare Dataset")
        except FileNotFoundError as e:
            print(f"Figshare: {e}")
    else:
        print("Figshare classification/yes folder not found. Run preprocess_figshare.py first.")

    # ---------- BraTS ----------
    brats_yes = r"D:\Brain\Datasets\BraTS\classification\yes"
    brats_no  = r"D:\Brain\Datasets\BraTS\classification\no"

    if os.path.exists(brats_yes):
        try:
            print("\nLoading BraTS dataset...")
            ds = TestDataset(brats_yes, brats_no)
            loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            test_model_on_dataset(model, loader, device, "BraTS Dataset")
        except FileNotFoundError as e:
            print(f"BraTS: {e}")
    else:
        print("\nBraTS classification/yes folder not found.")
        print("BraTS preprocessing is still running in the background. Run preprocess_brats.py first, then retry.")


if __name__ == "__main__":
    main()

