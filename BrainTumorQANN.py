import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create results directory
RESULTS_DIR = r"D:\Brain\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================== PREPROCESSING WITH VISUALIZATION =====================
def simple_preprocess(image, save_steps=False, filename=None):
    """Preprocessing with step visualization"""
    steps = {}
    
    # Step 0: Original
    image = np.clip(image, 0, 1)
    steps['0_original'] = image.copy()
    
    # Step 1: CLAHE
    img_uint = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_uint)
    steps['1_clahe'] = enhanced.astype(np.float32) / 255.0
    
    # Step 2: Normalization to [0,1]
    result = enhanced.astype(np.float32) / 255.0
    steps['2_normalized'] = result.copy()
    
    # Step 3: Normalization to [-1, 1]
    result = (result - 0.5) / 0.5
    steps['3_final'] = result.copy()
    
    # Save visualization if requested
    if save_steps and filename:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ['Original', 'After CLAHE', 'Normalized [0,1]', 'Final [-1,1]']
        
        for idx, (key, img) in enumerate(steps.items()):
            ax = axes[idx]
            if idx == 3:  # Final image in [-1, 1]
                display_img = (img + 1) / 2  # Convert to [0,1] for display
            else:
                display_img = img
            ax.imshow(display_img, cmap='gray')
            ax.set_title(titles[idx])
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'preprocessing_{filename}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    return result

def augment(image, mask=None):
    if random.random() > 0.5:
        image = TF.hflip(image)
        if mask is not None: mask = TF.hflip(mask)
    
    if random.random() > 0.5:
        image = TF.vflip(image)
        if mask is not None: mask = TF.vflip(mask)
    
    if random.random() > 0.4:
        angle = random.uniform(-20, 20)
        image = TF.rotate(image, angle, fill=0)
        if mask is not None: mask = TF.rotate(mask, angle, fill=0)
    
    return (image, mask) if mask is not None else image

# ===================== DATASETS =====================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True, size=(256,256), save_first=False):
        imgs = sorted(glob.glob(os.path.join(img_dir, '*.[pj][np]g')))
        masks = sorted(glob.glob(os.path.join(mask_dir, '*.[pj][np]g')))
        img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in imgs}
        mask_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in masks}
        common = sorted(set(img_dict.keys()) & set(mask_dict.keys()))
        self.pairs = [(img_dict[n], mask_dict[n]) for n in common]
        self.augment, self.size = augment, size
        self.save_first = save_first
        self.saved = False
        print(f"Loaded {len(self.pairs)} segmentation pairs")
    
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]
        img = Image.open(img_p).convert('L').resize(self.size, Image.BILINEAR)
        mask = Image.open(mask_p).convert('L').resize(self.size, Image.NEAREST)
        
        img = np.array(img, np.float32) / 255.0
        mask = (np.array(mask, np.uint8) > 127).astype(np.float32)
        
        # Save preprocessing steps for first image
        save_steps = self.save_first and not self.saved and idx == 0
        if save_steps:
            self.saved = True
        
        img = simple_preprocess(img, save_steps=save_steps, filename='sample_train')
        img = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.augment:
            img, mask = augment(img, mask)
        
        return img.float(), mask.float()

class ROIDataset(Dataset):
    def __init__(self, folder, split='train', size=(224,224)):
        yes_f = glob.glob(os.path.join(folder, 'yes', '*.[pj][np]g'))
        no_f = glob.glob(os.path.join(folder, 'no', '*.[pj][np]g'))
        
        x, y = yes_f + no_f, [1]*len(yes_f) + [0]*len(no_f)
        train_x, val_x, train_y, val_y = train_test_split(x, y, train_size=0.9, stratify=y, random_state=42)
        
        self.files, self.labels = (train_x, train_y) if split=='train' else (val_x, val_y)
        self.size, self.split = size, split
        print(f"{split.upper()}: {len(self.files)} images")
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L').resize(self.size, Image.BILINEAR)
        img = np.array(img, np.float32) / 255.0
        img = simple_preprocess(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        if self.split == 'train':
            img = augment(img)
        
        return img, int(self.labels[idx])

# ===================== ATTENTION UNET RESNET34 =====================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attention))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU()
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConvCBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)
    
    def forward(self, x):
        return self.cbam(self.conv(x))

class AttentionUNetResNet34(nn.Module):
    """Attention UNet with ResNet34 backbone, CBAM, and Attention Gates"""
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Encoder
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4
        
        # Attention gates
        self.att4 = AttentionGate(256, 256, 128)
        self.att3 = AttentionGate(128, 128, 64)
        self.att2 = AttentionGate(64, 64, 32)
        self.att1 = AttentionGate(64, 64, 32)
        
        # Decoder with CBAM
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConvCBAM(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConvCBAM(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConvCBAM(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = DoubleConvCBAM(128, 64)
        
        self.final = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d4 = self.up4(e4)
        e3_att = self.att4(d4, e3)
        d4 = self.dec4(torch.cat([d4, e3_att], dim=1))
        
        d3 = self.up3(d4)
        e2_att = self.att3(d3, e2)
        d3 = self.dec3(torch.cat([d3, e2_att], dim=1))
        
        d2 = self.up2(d3)
        e1_att = self.att2(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1_att], dim=1))
        
        d1 = self.up1(d2)
        e0_att = self.att1(d1, e0)
        d1 = self.dec1(torch.cat([d1, e0_att], dim=1))
        
        out = self.final(d1)
        return F.interpolate(out, size=size, mode='bilinear', align_corners=True)

# ===================== QUANTUM CLASSIFIER (UNCHANGED) =====================
try:
    import pennylane as qml
    n_qb = 10
    dev = qml.device('default.qubit', wires=n_qb)
    
    @qml.qnode(dev, interface='numpy')
    def quantum_circuit(inputs, weights):
        for i in range(n_qb):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)
        
        for l in range(5):
            for i in range(n_qb-1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n_qb-1, 0])
            
            for i in range(n_qb):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qb)]
    
    class QuantumLayer(nn.Module):
        def __init__(self, n_qb=10):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(5, n_qb, 3) * 0.05)
            self.n_qb = n_qb
        
        def forward(self, x):
            # CNN backbone runs on GPU. Only the tiny 10-value input
            # crosses to numpy for PennyLane, then result returns to GPU.
            device = x.device
            x_np = x.detach().cpu().numpy()          # shape: (batch, n_qb)
            w_np = self.weights.detach().cpu().numpy()  # shape: (5, n_qb, 3)
            res = np.stack([
                np.array(quantum_circuit(xi[:self.n_qb], w_np), dtype=np.float32)
                for xi in x_np
            ])  # shape: (batch, n_qb)
            return torch.from_numpy(res).to(device)  # back to GPU
    print("Using Quantum Layer (10 qubits, 5 layers)")
except:
    class QuantumLayer(nn.Module):
        def __init__(self, n_qb=10):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(n_qb, n_qb*4), nn.Tanh(),
                nn.Linear(n_qb*4, n_qb*2), nn.Tanh(),
                nn.Linear(n_qb*2, n_qb), nn.Tanh()
            )
        def forward(self, x):
            return self.fc(x[:,:10])
    print("Using classical fallback")

class QCNN(nn.Module):
    def __init__(self, n_qb=10, n_cls=2):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
        
        self.attention = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1024), nn.Sigmoid())
        
        self.to_quantum = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_qb)
        )
        
        self.quantum = QuantumLayer(n_qb)
        
        self.classical = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(n_qb + 64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_cls)
        )
    
    def forward(self, x):
        b = x.size(0)
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))).view(b, -1)
        
        att = self.attention(x)
        x_att = x * att
        
        q_out = self.quantum(self.to_quantum(x_att))
        c_out = self.classical(x_att)
        
        return self.classifier(torch.cat([q_out, c_out], dim=1))

# ===================== METRICS =====================
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1.0
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    smooth = 1.0
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.*inter + smooth) / (union + smooth) if union > 0 else torch.tensor(1.0)

def iou_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    smooth = 1.0
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth) if union > 0 else torch.tensor(1.0)

def psnr_score(pred, target):
    pred = torch.sigmoid(pred)
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ===================== VISUALIZATION FUNCTIONS =====================
def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Dice score
    axes[0, 0].plot(history['train_dice'], label='Train', marker='o')
    axes[0, 0].plot(history['val_dice'], label='Validation', marker='s')
    axes[0, 0].set_title('Dice Score Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU score
    axes[0, 1].plot(history['train_iou'], label='Train', marker='o')
    axes[0, 1].plot(history['val_iou'], label='Validation', marker='s')
    axes[0, 1].set_title('IoU Score Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    axes[1, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[1, 0].plot(history['val_loss'], label='Validation', marker='s')
    axes[1, 0].set_title('Loss Over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(history['val_dice'], label='Dice', marker='o')
    axes[1, 1].plot(history['val_iou'], label='IoU', marker='s')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, classes, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_classification_accuracy(models_acc, save_path):
    """Plot classification accuracy comparison"""
    plt.figure(figsize=(12, 6))
    models = list(models_acc.keys())
    accuracies = list(models_acc.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylim([90, 100])
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ===================== TRAINING =====================
def train_segmentation(model, train_dl, val_dl, device, epochs=40):
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    
    best_dice, best_iou, patience = 0.0, 0.0, 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'train_iou': [], 'val_iou': []}
    
    for ep in range(epochs):
        model.train()
        tr_loss, tr_dice, tr_iou = 0, 0, 0
        
        for img, mask in tqdm(train_dl, desc=f"Seg({ep+1}/{epochs})"):
            img, mask = img.to(device), mask.to(device)
            
            opt.zero_grad()
            out = model(img)
            loss = bce_dice_loss(out, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            tr_loss += loss.item()
            tr_dice += dice_score(out, mask).item()
            tr_iou += iou_score(out, mask).item()
        
        model.eval()
        va_loss, va_dice, va_iou = 0, 0, 0
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                
                va_loss += bce_dice_loss(out, mask).item()
                va_dice += dice_score(out, mask).item()
                va_iou += iou_score(out, mask).item()
        
        tr_loss /= len(train_dl)
        tr_dice /= len(train_dl)
        tr_iou /= len(train_dl)
        va_loss /= len(val_dl)
        va_dice /= len(val_dl)
        va_iou /= len(val_dl)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(va_dice)
        history['train_iou'].append(tr_iou)
        history['val_iou'].append(va_iou)
        
        scheduler.step(va_loss)
        
        print(f"Ep{ep+1}: TrDice={tr_dice:.4f} TrIoU={tr_iou:.4f} | VaDice={va_dice:.4f} VaIoU={va_iou:.4f}")
        
        if va_dice > best_dice:
            best_dice, best_iou, patience = va_dice, va_iou, 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_attention_unet.pth"))
            print(f"  *** BEST! Dice={best_dice:.4f} ***")
        else:
            patience += 1
            if patience >= 15:
                print("Early stop"); break
    
    # Plot training curves
    plot_training_curves(history, os.path.join(RESULTS_DIR, 'segmentation_training_curves.png'))
    
    print(f"\nBest Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"Best IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    return model

def extract_rois(model, img_dir, roi_dir, device):
    model.eval()
    os.makedirs(os.path.join(roi_dir, 'yes'), exist_ok=True)
    os.makedirs(os.path.join(roi_dir, 'no'), exist_ok=True)
    
    for cls in ['yes','no']:
        files = glob.glob(os.path.join(img_dir, cls, '*.[pj][np]g'))
        
        for i, f in enumerate(tqdm(files, desc=f"Extract {cls}")):
            img = Image.open(f).convert('L').resize((256,256), Image.BILINEAR)
            img = np.array(img, np.float32)/255.0
            img = simple_preprocess(img)
            tens = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                mask = (torch.sigmoid(model(tens))[0,0]>0.5).cpu().numpy()
            
            original = np.array(Image.open(f).convert('L').resize((256,256), Image.BILINEAR), np.float32)/255.0
            roi = original * mask if mask.sum() > 100 else original
            roi = cv2.resize((roi * 255).astype(np.uint8), (224,224))
            cv2.imwrite(os.path.join(roi_dir, cls, f"roi_{i:04d}.png"), roi)

def train_classification(model, train_dl, val_dl, device, epochs=50):
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)
    
    best_acc, best_metrics, patience = 0.0, None, 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0
        for img, lab in tqdm(train_dl, desc=f"Cls({ep+1}/{epochs})"):
            img, lab = img.to(device), lab.to(device)
            opt.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        
        model.eval()
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for img, lab in val_dl:
                img, lab = img.to(device), lab.to(device)
                out = model(img)
                labels.extend(lab.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                probs.extend(F.softmax(out,1).cpu().numpy())
        
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc_score = roc_auc_score(labels, np.array(probs)[:,1])
        
        history['train_loss'].append(tr_loss / len(train_dl))
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        
        scheduler.step(1-acc)
        
        print(f"Ep{ep+1}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc_score:.4f}")
        
        if acc > best_acc:
            best_acc, patience = acc, 0
            cm = confusion_matrix(labels, preds)
            best_metrics = {
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc_score,
                'cm': cm, 'labels': labels, 'preds': preds, 'probs': np.array(probs)[:,1]
            }
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_qcnn.pth"))
            print(f"  *** BEST! Acc={acc*100:.2f}% ***")
        else:
            patience += 1
            if patience >= 15:
                print("Early stop"); break
    
    # Save final results
    tn, fp, fn, tp = best_metrics['cm'].ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Plot confusion matrix
    plot_confusion_matrix(best_metrics['cm'], ['No Tumor', 'Tumor'], 
                         os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plot_roc_curve(best_metrics['labels'], best_metrics['probs'],
                  os.path.join(RESULTS_DIR, 'roc_curve.png'))
    
    # Plot model comparison
    models_acc = {'QCNN (Ours)': best_metrics['acc'] * 100}
    plot_classification_accuracy(models_acc, os.path.join(RESULTS_DIR, 'model_comparison.png'))
    
    print(f"\n{'='*70}\nFINAL CLASSIFICATION RESULTS\n{'='*70}")
    print(f"Accuracy:      {best_metrics['acc']:.4f} ({best_metrics['acc']*100:.2f}%)")
    print(f"Precision:     {best_metrics['prec']:.4f}")
    print(f"Recall:        {best_metrics['rec']:.4f}")
    print(f"Specificity:   {spec:.4f}")
    print(f"F1-Score:      {best_metrics['f1']:.4f}")
    print(f"NPV:           {npv:.4f}")
    print(f"AUC-ROC:       {best_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:\n{best_metrics['cm']}")
    print(f"\n{classification_report(best_metrics['labels'], best_metrics['preds'], digits=4, target_names=['No Tumor', 'Tumor'])}")
    
    return model

# ===================== MAIN =====================
def main():
    config = {
        'train_seg_img': r"D:\archive (2)\Br35H-Mask-RCNN\TRAIN",
        'train_seg_mask': r"D:\archive (2)\Br35H-Mask-RCNN\TRAIN_MASKS",
        'val_seg_img': r"D:\archive (2)\Br35H-Mask-RCNN\VAL",
        'val_seg_mask': r"D:\archive (2)\Br35H-Mask-RCNN\VAL_MASKS",
        'orig_img': r"D:\archive (2)",
        'roi_out': r"D:\archive (2)\roi_attention_unet",
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    print("="*70)
    print("ADVANCED PIPELINE: AttentionUNetResNet34 + QCNN")
    print("="*70)
    print(f"Device: {config['device']}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")
    
    print("STAGE 1: SEGMENTATION")
    tr_ds = SegDataset(config['train_seg_img'], config['train_seg_mask'], size=(256,256), save_first=True)
    va_ds = SegDataset(config['val_seg_img'], config['val_seg_mask'], False, (256,256))
    tr_dl = DataLoader(tr_ds, 8, True, num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, 8, False, num_workers=0, pin_memory=True)
    
    seg_model = AttentionUNetResNet34().to(config['device'])
    train_segmentation(seg_model, tr_dl, va_dl, config['device'], 40)
    
    # Load best model
    if os.path.exists(os.path.join(RESULTS_DIR, "best_attention_unet.pth")):
        seg_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_attention_unet.pth")))
        print("Loaded best segmentation model")
    
    print("\nSTAGE 2: ROI EXTRACTION")
    extract_rois(seg_model, config['orig_img'], config['roi_out'], config['device'])
    
    print("\nSTAGE 3: QCNN CLASSIFICATION")
    tr_cls = ROIDataset(config['roi_out'], 'train', (224,224))
    va_cls = ROIDataset(config['roi_out'], 'val', (224,224))
    tr_dl_cls = DataLoader(tr_cls, 32, True, num_workers=0, pin_memory=True)
    va_dl_cls = DataLoader(va_cls, 32, False, num_workers=0, pin_memory=True)
    
    qcnn = QCNN(10, 2).to(config['device'])
    train_classification(qcnn, tr_dl_cls, va_dl_cls, config['device'], 50)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print(f"Models saved:")
    print(f"  - {os.path.join(RESULTS_DIR, 'best_attention_unet.pth')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'best_qcnn.pth')}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
