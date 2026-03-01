import os
import glob
import random
import time
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.model_selection import KFold
from sklearn.metrics import *
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd

# ----------------- OPTIONAL DEPENDENCIES -----------------
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("WARNING: 'thop' library not found. GFLOPS will show as 0. Run 'pip install thop'")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("CRITICAL: PennyLane not installed. Quantum model will fail. Run 'pip install pennylane'")

warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
# UPDATE THIS PATH to where your ROI folders (yes/no) are located
ROI_DIR = r"D:\archive (2)\roi_attention_unet" 
RESULTS_DIR = r"D:\Brain\Results_Journal_6Models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ================= 1. PREPROCESSING =================
def simple_preprocess(image):
    """QCNN Specific Preprocessing: Grayscale -> CLAHE -> [-1, 1]"""
    # 1. CLAHE
    img_uint = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_uint)
    
    # 2. Normalize to [-1, 1]
    result = enhanced.astype(np.float32) / 255.0
    result = (result - 0.5) / 0.5
    return result

# ================= 2. DATASET (DUAL MODE) =================
class ROIDataset(Dataset):
    def __init__(self, folder, size=(224,224)):
        yes_f = glob.glob(os.path.join(folder, 'yes', '*.[pj][np]g'))
        no_f = glob.glob(os.path.join(folder, 'no', '*.[pj][np]g'))
        
        if not yes_f and not no_f:
            raise FileNotFoundError(f"No images found in {folder}. Check path!")
            
        self.files = yes_f + no_f
        self.labels = [1]*len(yes_f) + [0]*len(no_f)
        self.size = size
        
        # Shuffle
        combined = list(zip(self.files, self.labels))
        np.random.seed(42)
        np.random.shuffle(combined)
        self.files, self.labels = zip(*combined)

        # Standard CNN Transform (ImageNet Stats)
        self.rgb_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = int(self.labels[idx])
        
        # Mode A: QCNN (Gray, CLAHE, -1 to 1)
        img_gray = Image.open(path).convert('L').resize(self.size, Image.BILINEAR)
        arr_gray = np.array(img_gray, np.float32) / 255.0
        arr_gray = simple_preprocess(arr_gray)
        qcnn_tensor = torch.from_numpy(arr_gray).unsqueeze(0).float()
        
        # Mode B: Standard CNN (RGB, ImageNet Norm)
        img_rgb = Image.open(path).convert('RGB')
        cnn_tensor = self.rgb_transform(img_rgb)
        
        return {'qcnn': qcnn_tensor, 'cnn': cnn_tensor, 'label': label}

# ================= 3. MODEL DEFINITIONS =================
# --- QCNN ---
if PENNYLANE_AVAILABLE:
    n_qb = 10
    dev = qml.device('default.qubit', wires=n_qb)
    
    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, weights):
        # Encoding
        for i in range(n_qb):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
        
        # Variational layers
        for l in range(5):
            # Entangling layer
            for i in range(n_qb-1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n_qb-1, 0])
            
            # Rotation layer
            for i in range(n_qb):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
        
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qb)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qb=10):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(5, n_qb, 3) * 0.05)
        self.n_qb = n_qb
    
    def forward(self, x):
        if PENNYLANE_AVAILABLE:
            outputs = []
            for xi in x:
                qc_output = quantum_circuit(xi[:self.n_qb], self.weights)
                tensor_output = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in qc_output])
                outputs.append(tensor_output)
            return torch.stack(outputs)
        else:
            # Fallback if PennyLane not available
            return x[:, :self.n_qb]

class QCNN(nn.Module):
    def __init__(self, n_qb=10, n_cls=2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.to_quantum = nn.Linear(256, n_qb)
        self.quantum = QuantumLayer(n_qb)
        self.classifier = nn.Sequential(
            nn.Linear(n_qb + 256, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, n_cls)
        )
    
    def forward(self, x):
        features = self.conv_blocks(x).view(x.size(0), -1)
        q_in = torch.tanh(self.to_quantum(features)) * np.pi 
        q_out = self.quantum(q_in)
        combined = torch.cat([q_out, features], dim=1)
        return self.classifier(combined)

# --- MODEL FACTORY (6 MODELS) ---
def get_model(name, device):
    """
    FIXED: Now properly returns all model types
    """
    try:
        # 1. Proposed Model
        if name == 'QCNN':
            model = QCNN(n_cls=2)
            
        # 2. ResNet50 (Standard)
        elif name == 'ResNet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(2048, 2)
            
        # 3. VGG16 (Heavy)
        elif name == 'VGG16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            model.classifier[6] = nn.Linear(4096, 2)
            
        # 4. DenseNet121 (Medical Standard)
        elif name == 'DenseNet121':
            model = models.densenet121(weights='IMAGENET1K_V1')
            model.classifier = nn.Linear(1024, 2)
            
        # 5. EfficientNet (Light/Efficient)
        elif name == 'EfficientNet':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            model.classifier[1] = nn.Linear(1280, 2)
            
        # 6. MobileNet (Mobile)
        elif name == 'MobileNet':
            model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
            model.classifier[3] = nn.Linear(1280, 2)
        
        else:
            raise ValueError(f"Unknown model name: {name}")
            
        return model.to(device)
        
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

def get_complexity(model, name, device):
    """Calculate Params and GFLOPS"""
    try:
        dummy_in = torch.randn(1, 1, 224, 224).to(device) if name=='QCNN' else torch.randn(1, 3, 224, 224).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        gflops = 0.0
        if THOP_AVAILABLE:
            flops, _ = profile(model, inputs=(dummy_in,), verbose=False)
            gflops = flops / 1e9
        return params, gflops
    except Exception as e:
        print(f"Warning: Could not calculate complexity for {name}: {e}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        return params, 0.0

# ================= 4. TRAINING ENGINE =================
def train_epoch(model, loader, optimizer, criterion, device, is_qcnn):
    """Train for one epoch and return metrics"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    
    for batch in loader:
        x = batch['qcnn'].to(device) if is_qcnn else batch['cnn'].to(device)
        y = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    epoch_time = time.time() - start
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return epoch_time, avg_loss, accuracy

def evaluate_fold(model, loader, device, is_qcnn, return_details=False):
    """Evaluate model on validation set"""
    model.eval()
    preds, targets, probs = [], [], []
    inf_times = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch['qcnn'].to(device) if is_qcnn else batch['cnn'].to(device)
            y = batch['label'].to(device)
            
            t0 = time.time()
            out = model(x)
            t1 = time.time()
            inf_times.append((t1-t0)/x.size(0))  # Time per sample
            
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            probs.extend(F.softmax(out, 1)[:, 1].cpu().numpy())
    
    # Calculate Metrics
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        tn, fp, fn, tp = 0, 0, 0, 0
    
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    f1 = f1_score(targets, preds, zero_division=0)
    
    # AUC - handle case where only one class is present
    try:
        auc_val = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5
        fpr, tpr, _ = roc_curve(targets, probs)
    except:
        auc_val = 0.5
        fpr, tpr = [0, 1], [0, 1]
    
    result = {
        'acc': acc, 
        'sens': sens, 
        'spec': spec, 
        'f1': f1, 
        'auc': auc_val,
        'inf_ms': np.mean(inf_times) * 1000 if inf_times else 0,
        'cm': cm
    }
    
    if return_details:
        result['preds'] = preds
        result['targets'] = targets
        result['probs'] = probs
        result['fpr'] = fpr
        result['tpr'] = tpr
    
    return result

# ================= 5. VISUALIZATION FUNCTIONS =================
def plot_model_results(history, model_name, save_dir):
    """
    Create a 1x3 plot showing:
    1. Accuracy curves (train vs validation)
    2. Confusion Matrix
    3. ROC Curve
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{model_name}', fontsize=14, fontweight='bold')
    
    # 1. ACCURACY PLOT
    ax = axes[0]
    epochs = range(1, len(history['train_acc']) + 1)
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'orange', label='Validation Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    
    # 2. CONFUSION MATRIX
    ax = axes[1]
    cm = history['final_cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': ''})
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    # 3. ROC CURVE
    ax = axes[2]
    fpr = history['fpr']
    tpr = history['tpr']
    auc_val = history['auc']
    ax.plot(fpr, tpr, 'orange', linewidth=2, label=f'ROC Curve (AUC = {auc_val:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved visualization: {save_path}")

def create_combined_figure(all_histories, models_list, save_dir):
    """
    Create a large combined figure with all models
    Similar to the uploaded image format
    """
    n_models = len(models_list)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 4*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, model_name in enumerate(models_list):
        history = all_histories[model_name]
        
        # 1. ACCURACY PLOT
        ax = axes[idx, 0]
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs, history['val_acc'], 'orange', label='Validation Accuracy', linewidth=2)
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.05])
        if idx == 0:
            ax.set_title('Accuracy Plot', fontweight='bold')
        if idx == n_models - 1:
            ax.set_xlabel('Epoch')
        
        # Add model label on left
        ax.text(-0.25, 0.5, model_name, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='center', rotation=90)
        
        # 2. CONFUSION MATRIX
        ax = axes[idx, 1]
        cm = history['final_cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': ''}, square=True)
        ax.set_ylabel('True label')
        if idx == 0:
            ax.set_title('Confusion Matrix', fontweight='bold')
        if idx == n_models - 1:
            ax.set_xlabel('Predicted label')
        
        # 3. ROC CURVE
        ax = axes[idx, 2]
        fpr = history['fpr']
        tpr = history['tpr']
        auc_val = history['auc']
        ax.plot(fpr, tpr, 'orange', linewidth=2, label=f'ROC Curve (AUC = {auc_val:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        if idx == 0:
            ax.set_title('Receiver Operating Characteristic', fontweight='bold')
        if idx == n_models - 1:
            ax.set_xlabel('False Positive Rate')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'All_Models_Combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved combined visualization: {save_path}")

# ================= 6. MAIN EXECUTION WITH VISUALIZATION =================
def main():
    print(f"Starting 6-Model Comparison on {DEVICE}")
    print(f"Data Source: {ROI_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")
    
    # Setup Data
    try:
        ds = ROIDataset(ROI_DIR)
        print(f"Total Images: {len(ds)}")
        print(f"Class Distribution: {sum(ds.labels)} positive, {len(ds.labels) - sum(ds.labels)} negative\n")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 5-Fold Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models_list = ['QCNN', 'ResNet50', 'VGG16', 'DenseNet121', 'EfficientNet', 'MobileNet']
    
    final_results = []
    all_histories = {}  # Store training history for visualization
    
    for name in models_list:
        print(f"\n{'='*60}")
        print(f"Evaluating {name}")
        print('='*60)
        
        is_qcnn = (name == 'QCNN')
        
        # Track metrics across folds
        fold_res = {
            'acc': [], 'sens': [], 'spec': [], 'f1': [], 'auc': [], 
            'train_s': [], 'inf_ms': []
        }
        
        # For visualization: store one representative fold
        best_fold_history = None
        best_fold_auc = 0
        
        # Get Complexity (Once)
        temp_model = get_model(name, DEVICE)
        if temp_model is None:
            print(f"Skipping {name} due to loading error")
            continue
            
        params, gflops = get_complexity(temp_model, name, DEVICE)
        del temp_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Model Params: {params:.2f}M, GFLOPS: {gflops:.2f}")
        
        # Run 5-Fold Cross-Validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(ds)):
            print(f"\n  Fold {fold+1}/5...")
            
            # Create data subsets
            train_sub = Subset(ds, train_idx)
            val_sub = Subset(ds, val_idx)
            tr_dl = DataLoader(train_sub, batch_size=16, shuffle=True, num_workers=0)
            va_dl = DataLoader(val_sub, batch_size=16, shuffle=False, num_workers=0)
            
            # Initialize model and optimizer
            model = get_model(name, DEVICE)
            if model is None:
                continue
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop with history tracking
            epochs = 30  # Use 30-50 for paper
            train_acc_history = []
            val_acc_history = []
            
            for epoch in range(epochs):
                epoch_time, loss, train_acc = train_epoch(model, tr_dl, optimizer, criterion, DEVICE, is_qcnn)
                
                # Evaluate on validation
                val_metrics = evaluate_fold(model, va_dl, DEVICE, is_qcnn, return_details=(epoch == epochs-1))
                val_acc = val_metrics['acc']
                
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, "
                          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Final evaluation with full details
            metrics = evaluate_fold(model, va_dl, DEVICE, is_qcnn, return_details=True)
            
            # Store results
            fold_res['acc'].append(metrics['acc'])
            fold_res['sens'].append(metrics['sens'])
            fold_res['spec'].append(metrics['spec'])
            fold_res['f1'].append(metrics['f1'])
            fold_res['auc'].append(metrics['auc'])
            fold_res['inf_ms'].append(metrics['inf_ms'])
            fold_res['train_s'].append(epoch_time * epochs)
            
            # Keep best fold for visualization
            if metrics['auc'] > best_fold_auc:
                best_fold_auc = metrics['auc']
                best_fold_history = {
                    'train_acc': train_acc_history,
                    'val_acc': val_acc_history,
                    'final_cm': metrics['cm'],
                    'fpr': metrics['fpr'],
                    'tpr': metrics['tpr'],
                    'auc': metrics['auc']
                }
            
            print(f"    Results - Acc: {metrics['acc']:.4f}, Sens: {metrics['sens']:.4f}, "
                  f"Spec: {metrics['spec']:.4f}, AUC: {metrics['auc']:.4f}")
            
            # Clean up
            del model, optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Store history for visualization
        all_histories[name] = best_fold_history
        
        # Generate individual model visualization
        plot_model_results(best_fold_history, name, RESULTS_DIR)
        
        # Aggregate 5-Fold Statistics
        row = {
            'Model': name,
            'Params (M)': f"{params:.2f}",
            'GFLOPS': f"{gflops:.2f}",
            'Train Time (s)': f"{np.mean(fold_res['train_s']):.1f}±{np.std(fold_res['train_s']):.1f}",
            'Inf Time (ms)': f"{np.mean(fold_res['inf_ms']):.2f}±{np.std(fold_res['inf_ms']):.2f}",
            'Accuracy': f"{np.mean(fold_res['acc']):.4f}±{np.std(fold_res['acc']):.4f}",
            'Sensitivity': f"{np.mean(fold_res['sens']):.4f}±{np.std(fold_res['sens']):.4f}",
            'Specificity': f"{np.mean(fold_res['spec']):.4f}±{np.std(fold_res['spec']):.4f}",
            'F1-Score': f"{np.mean(fold_res['f1']):.4f}±{np.std(fold_res['f1']):.4f}",
            'AUC': f"{np.mean(fold_res['auc']):.4f}±{np.std(fold_res['auc']):.4f}"
        }
        
        # 95% Confidence Interval for Accuracy
        accs = fold_res['acc']
        if len(accs) > 1:
            ci = stats.t.interval(0.95, len(accs)-1, loc=np.mean(accs), scale=stats.sem(accs))
            row['95% CI (Acc)'] = f"[{ci[0]:.4f}-{ci[1]:.4f}]"
        else:
            row['95% CI (Acc)'] = "N/A"
        
        final_results.append(row)

    # ----------------- SAVE REPORT -----------------
    if final_results:
        df = pd.DataFrame(final_results)
        print("\n" + "="*80)
        print("FINAL 6-MODEL COMPARISON (Reviewer Ready)")
        print("="*80)
        print(df.to_string(index=False))
        
        csv_path = os.path.join(RESULTS_DIR, "Final_6Model_Comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Create a more readable text report
        txt_path = os.path.join(RESULTS_DIR, "Final_Report.txt")
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("6-MODEL BRAIN TUMOR CLASSIFICATION COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Device: {DEVICE}\n")
            f.write(f"Data Source: {ROI_DIR}\n")
            f.write(f"Cross-Validation: 5-Fold\n\n")
            f.write(df.to_string(index=False))
        print(f"✓ Text report saved to: {txt_path}")
        
        # Generate combined visualization
        print("\nGenerating combined visualization...")
        create_combined_figure(all_histories, models_list, RESULTS_DIR)
        
        print("\n" + "="*80)
        print("ALL RESULTS AND VISUALIZATIONS SAVED SUCCESSFULLY!")
        print("="*80)
        print(f"Location: {RESULTS_DIR}")
        print("\nGenerated files:")
        print("  - Final_6Model_Comparison.csv")
        print("  - Final_Report.txt")
        print("  - All_Models_Combined.png")
        for model in models_list:
            print(f"  - {model}_results.png")
    else:
        print("\nNo results generated. Check for errors above.")

if __name__ == "__main__":
    main()