import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import glob
from PIL import Image

# =========================
# CONFIG
# =========================
RESULTS_DIR = r"D:\Brain\Results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("TRAINING ALL BASELINE MODELS (5-FOLD CV)")
print("="*70)
print(f"Device: {device}")
print(f"Results: {RESULTS_DIR}\n")

# =========================
# IMPORT FUNCTIONS
# =========================
from BrainTumorQANN_FAST import simple_preprocess, augment, QCNN

# =========================
# DATASET
# =========================
class KFoldROIDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, augment_flag=False, size=(224,224)):
        self.files = files
        self.labels = labels
        self.size = size
        self.augment_flag = augment_flag
        
    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L').resize(self.size, Image.BILINEAR)
        img = np.array(img, np.float32) / 255.0
        img = simple_preprocess(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        if self.augment_flag:
            img = augment(img)
            
        return img, int(self.labels[idx])

# =========================
# MODEL DEFINITIONS
# =========================
def get_model(name):
    if name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif name == "QCNN":
        model = QCNN(10, 2)
    
    return model.to(device)

# =========================
# COMPLEXITY CALCULATION
# =========================
def compute_complexity(model):
    from thop import profile
    
    dummy = torch.randn(1, 1, 224, 224).to(device)
    
    # Parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # GFLOPs
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    gflops = (flops * 2) / 1e9
    
    # Inference time
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inf_time = ((time.time() - start) / 100) * 1000
    
    return params, gflops, inf_time

# =========================
# 5-FOLD TRAINING
# =========================
def train_model_5fold(files, labels, model_name, epochs=25, k_folds=5):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []
    fold_times = []
    
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name} (5-FOLD CV)")
    print(f"{'='*70}\n")
    
    total_start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        
        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        tr_ds = KFoldROIDataset(train_files, train_labels, augment_flag=True)
        va_ds = KFoldROIDataset(val_files, val_labels, augment_flag=False)
        
        tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        model = get_model(model_name)
        opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5, factor=0.5)
        scaler = GradScaler()
        
        best_acc = 0.0
        patience_counter = 0
        best_metrics = {}
        fold_start = time.time()
        epoch_times = []
        
        for ep in range(epochs):
            epoch_start = time.time()
            
            # TRAIN
            model.train()
            tr_loss = 0
            
            for img, lab in tqdm(tr_dl, desc=f"Fold {fold+1} Ep {ep+1}", leave=False):
                img, lab = img.to(device), lab.to(device)
                opt.zero_grad()
                
                with autocast():
                    out = model(img)
                    loss = F.cross_entropy(out, lab)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tr_loss += loss.item()
            
            # VALIDATION
            model.eval()
            preds, true_labels, probs = [], [], []
            
            with torch.no_grad():
                for img, lab in va_dl:
                    img, lab = img.to(device), lab.to(device)
                    out = model(img)
                    true_labels.extend(lab.cpu().numpy())
                    preds.extend(out.argmax(1).cpu().numpy())
                    probs.extend(F.softmax(out, 1).cpu().numpy())
            
            acc = accuracy_score(true_labels, preds)
            scheduler.step(acc)
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            print(f"  Ep {ep+1}: Loss={tr_loss/len(tr_dl):.4f}, Acc={acc:.4f}, Time={epoch_time:.1f}s")
            
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                
                best_metrics = {
                    'acc': acc,
                    'prec': precision_score(true_labels, preds, zero_division=0),
                    'rec': recall_score(true_labels, preds, zero_division=0),
                    'f1': f1_score(true_labels, preds, zero_division=0),
                    'auc': roc_auc_score(true_labels, np.array(probs)[:, 1])
                }
                
                torch.save(model.state_dict(), 
                          os.path.join(RESULTS_DIR, f"{model_name}_fold_{fold+1}.pth"))
                print(f"    ✓ Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stop at epoch {ep+1}")
                    break
        
        fold_time = time.time() - fold_start
        avg_epoch_time = np.mean(epoch_times)
        
        best_metrics['fold_time'] = fold_time
        best_metrics['avg_epoch_time'] = avg_epoch_time
        
        fold_metrics.append(best_metrics)
        fold_times.append(fold_time)
        
        print(f"  Fold {fold+1}: Best Acc={best_acc*100:.2f}%, Time={fold_time/60:.2f}min")
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*70}")
    print(f"{model_name} - FINAL RESULTS")
    print(f"{'='*70}")
    
    acc_scores = [f['acc'] for f in fold_metrics]
    prec_scores = [f['prec'] for f in fold_metrics]
    rec_scores = [f['rec'] for f in fold_metrics]
    f1_scores = [f['f1'] for f in fold_metrics]
    auc_scores = [f['auc'] for f in fold_metrics]
    epoch_times = [f['avg_epoch_time'] for f in fold_metrics]
    
    print(f"Accuracy:  {np.mean(acc_scores)*100:.1f} ± {np.std(acc_scores)*100:.1f}%")
    print(f"Precision: {np.mean(prec_scores)*100:.1f} ± {np.std(prec_scores)*100:.1f}%")
    print(f"Recall:    {np.mean(rec_scores)*100:.1f} ± {np.std(rec_scores)*100:.1f}%")
    print(f"F1-Score:  {np.mean(f1_scores)*100:.1f} ± {np.std(f1_scores)*100:.1f}%")
    print(f"AUC:       {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"Avg Epoch Time: {np.mean(epoch_times)/60:.2f} min")
    print(f"Total Training Time: {total_time/3600:.2f} hours")
    print("="*70)
    
    return fold_metrics, total_time

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Load dataset
    roi_folder = r"D:\archive (2)\roi_attention_unet"
    
    print("Loading ROI dataset...")
    yes_files = glob.glob(os.path.join(roi_folder, 'yes', '*.[pj][np]g'))
    no_files = glob.glob(os.path.join(roi_folder, 'no', '*.[pj][np]g'))
    all_files = yes_files + no_files
    all_labels = [1]*len(yes_files) + [0]*len(no_files)
    
    print(f"Total: {len(all_files)} images (Yes: {len(yes_files)}, No: {len(no_files)})\n")
    
    # Train all models
    models_to_train = ["ResNet18", "ResNet50", "VGG16", "DenseNet121"]
    all_results = {}
    
    script_start = time.time()
    
    for model_name in models_to_train:
        fold_metrics, total_time = train_model_5fold(
            all_files, all_labels, model_name, epochs=25, k_folds=5
        )
        all_results[model_name] = {
            'metrics': fold_metrics,
            'total_time': total_time
        }
    
    # Also evaluate existing QCNN
    print(f"\n{'='*70}")
    print("EVALUATING EXISTING QCNN MODELS")
    print(f"{'='*70}")
    
    from BrainTumorQANN_FAST import ROIDataset
    val_dataset = ROIDataset(roi_folder, split='val', size=(224,224))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    qcnn_metrics = []
    for fold in range(1, 6):
        model = QCNN(10, 2).to(device)
        model.load_state_dict(torch.load(
            os.path.join(RESULTS_DIR, f"qcnn_fold_{fold}.pth"), 
            map_location=device
        ))
        
        model.eval()
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for img, lab in val_loader:
                img, lab = img.to(device), lab.to(device)
                out = model(img)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(lab.cpu().numpy())
                probs.extend(F.softmax(out, 1)[:,1].cpu().numpy())
        
        qcnn_metrics.append({
            'acc': accuracy_score(labels, preds),
            'prec': precision_score(labels, preds),
            'rec': recall_score(labels, preds),
            'f1': f1_score(labels, preds),
            'auc': roc_auc_score(labels, probs)
        })
    
    all_results['QCNN'] = {'metrics': qcnn_metrics, 'total_time': 0}
    
    # =========================
    # GENERATE TABLES
    # =========================
    print(f"\n{'='*70}")
    print("GENERATING PUBLICATION TABLES")
    print(f"{'='*70}\n")
    
    table1_data = []
    table2_data = []
    
    all_models = ["ResNet18", "ResNet50", "VGG16", "DenseNet121", "QCNN"]
    
    for model_name in all_models:
        metrics = all_results[model_name]['metrics']
        
        acc_scores = [m['acc'] for m in metrics]
        prec_scores = [m['prec'] for m in metrics]
        rec_scores = [m['rec'] for m in metrics]
        f1_scores = [m['f1'] for m in metrics]
        auc_scores = [m['auc'] for m in metrics]
        
        # Complexity
        model = get_model(model_name)
        params, gflops, inf_time = compute_complexity(model)
        
        # TABLE 1
        display_name = "Proposed DAU-HQCNN" if model_name == "QCNN" else model_name
        
        table1_data.append({
            'Model': display_name,
            'Parameters (M)': f"{params:.1f}",
            'Accuracy (%)': f"{np.mean(acc_scores)*100:.1f} ± {np.std(acc_scores)*100:.1f}",
            'Precision (%)': f"{np.mean(prec_scores)*100:.1f} ± {np.std(prec_scores)*100:.1f}",
            'Recall (%)': f"{np.mean(rec_scores)*100:.1f} ± {np.std(rec_scores)*100:.1f}",
            'F1-Score (%)': f"{np.mean(f1_scores)*100:.1f} ± {np.std(f1_scores)*100:.1f}",
            'AUC': f"{np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}"
        })
        
        # TABLE 2
        total_time_hrs = all_results[model_name]['total_time'] / 3600 if all_results[model_name]['total_time'] > 0 else 0
        avg_epoch_time = np.mean([m['avg_epoch_time'] for m in metrics if 'avg_epoch_time' in m]) / 60 if any('avg_epoch_time' in m for m in metrics) else 0
        
        table2_data.append({
            'Model': display_name,
            'Trainable Parameters (M)': f"{params:.1f}",
            'GFLOPs (224×224)': f"{gflops:.1f}",
            'Training Time / Epoch (min)': f"{avg_epoch_time:.1f}" if avg_epoch_time > 0 else '--',
            'Total Training Time (5-Fold) (hrs)': f"{total_time_hrs:.1f}" if total_time_hrs > 0 else '--',
            'Inference Time (ms/image)': f"{inf_time:.1f}"
        })
    
    df1 = pd.DataFrame(table1_data)
    df2 = pd.DataFrame(table2_data)
    
    print("TABLE 1: PERFORMANCE COMPARISON")
    print(df1.to_string(index=False))
    
    print(f"\n{'='*70}\n")
    
    print("TABLE 2: COMPUTATIONAL COMPLEXITY")
    print(df2.to_string(index=False))
    
    # Save
    df1.to_excel(os.path.join(RESULTS_DIR, "Journal_Table1_Performance.xlsx"), index=False)
    df2.to_excel(os.path.join(RESULTS_DIR, "Journal_Table2_Computational.xlsx"), index=False)
    df1.to_csv(os.path.join(RESULTS_DIR, "Journal_Table1_Performance.csv"), index=False)
    df2.to_csv(os.path.join(RESULTS_DIR, "Journal_Table2_Computational.csv"), index=False)
    
    total_script_time = time.time() - script_start
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_script_time/3600:.2f} hours")
    print(f"\nTables saved:")
    print(f"  - Journal_Table1_Performance.xlsx")
    print(f"  - Journal_Table2_Computational.xlsx")
    print(f"{'='*70}")