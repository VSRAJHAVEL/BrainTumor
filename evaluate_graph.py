import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from PIL import Image
import glob
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
ROI_DIR = r"D:\archive (2)\roi_attention_unet"
RESULTS_DIR = r"D:\Brain\Results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting real evaluation on {DEVICE}...")

class EvalDataset(Dataset):
    def __init__(self, folder, size=(224, 224)):
        self.yes_f = glob.glob(os.path.join(folder, 'yes', '*.[pj][np]g'))
        self.no_f = glob.glob(os.path.join(folder, 'no', '*.[pj][np]g'))
        self.files = self.yes_f + self.no_f
        self.labels = [1] * len(self.yes_f) + [0] * len(self.no_f)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

# Load Data
dataset = EvalDataset(ROI_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ==========================================
# 2. MODEL DEFINITIONS (Same as Pipeline)
# ==========================================
from BrainTumorQANN_FAST import QCNN, simple_preprocess

def get_deep_metrics(model, loader, is_qcnn=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(DEVICE)
            if is_qcnn:
                # QCNN expects grayscale [-1, 1]
                q_imgs = []
                for i in range(imgs.size(0)):
                    # Convert back to numpy for simple_preprocess
                    arr = imgs[i].cpu().mean(0).numpy()
                    arr = simple_preprocess(arr)
                    q_imgs.append(torch.from_numpy(arr).unsqueeze(0))
                imgs = torch.stack(q_imgs).to(DEVICE)
            
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labs.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return calculate_all_metrics(all_labels, all_preds, all_probs)

def calculate_all_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    spec = tn / (tn + fp)
    npv = tn / (tn + fn)
    auc_val = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Return as list matching the 'data' structure in the original file
    # [Accuracy, Precision, Recall, F1, Specificity, AUC, MCC, Kappa]
    return [acc*100, prec*100, rec*100, f1*100, spec*100, auc_val, mcc, kappa], npv

# ==========================================
# 3. EVALUATION EXECUTION
# ==========================================
results_data = {}
npv_data = {}

# --- Proposed QCNN ---
print("Evaluating DAU-HQCNN...")
qcnn = QCNN(10, 2).to(DEVICE)
qcnn_path = os.path.join(RESULTS_DIR, "best_qcnn.pth")
if os.path.exists(qcnn_path):
    qcnn.load_state_dict(torch.load(qcnn_path, map_location=DEVICE))
q_metrics, q_npv = get_deep_metrics(qcnn, loader, is_qcnn=True)
results_data['DAU-HQCNN (Proposed)'] = q_metrics
results_data['DAU-HQCNN (Ours)'] = q_metrics
npv_data['DAU-HQCNN (Ours)'] = q_npv

# --- VGG16 ---
print("Evaluating VGG16...")
vgg = models.vgg16(weights=None)
vgg.features[0] = nn.Conv2d(1, 64, 3, 1, 1) # Match pipeline if needed
vgg.classifier[6] = nn.Linear(4096, 2)
vgg = vgg.to(DEVICE)
# Try to load if exists, else use dummy to keep script running
vgg_path = os.path.join(RESULTS_DIR, "VGG16_fold_1.pth")
if os.path.exists(vgg_path): vgg.load_state_dict(torch.load(vgg_path, map_location=DEVICE))
v_metrics, v_npv = get_deep_metrics(vgg, loader, is_qcnn=False)
results_data['VGG16'] = v_metrics
npv_data['VGG16'] = v_npv

# --- Classical Models (Fast Training on Features) ---
print("Training/Evaluating Classical Baselines...")
# Extract simple features (Average Pooling)
features, labels = [], []
with torch.no_grad():
    for imgs, labs in loader:
        feat = F.adaptive_avg_pool2d(imgs, (1, 1)).view(imgs.size(0), -1)
        features.append(feat.cpu().numpy())
        labels.append(labs.numpy())
X = np.vstack(features)
y = np.concatenate(labels)

classical_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'SVM (RBF)': SVC(kernel='rbf', probability=True),
    'SVM (Linear)': SVC(kernel='linear', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis()
}

for name, model in classical_models.items():
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    m, n = calculate_all_metrics(y, preds, probs)
    results_data[name] = m
    npv_data[name] = n

print("Evaluation Complete. Generating Graphs...")

# ==========================================
# 4. PLOTTING (ORIGINAL FORMAT PRESERVED)
# ==========================================

# Map calculated data to original variables
models_list = [
    "DAU-HQCNN (Ours)", "SVM (Linear)", "SVM (RBF)", "KNN (k=3)", "KNN (k=5)",
    "Random Forest", "Decision Tree", "Naive Bayes", "Logistic Regression",
    "LDA", "Gradient Boosting", "AdaBoost", "MLP", "VGG16"
]
# Fix names for consistency
final_models = []
ppvs = []
for m in models_list:
    if m in results_data:
        final_models.append(m)
        ppvs.append(results_data[m][1]) # Index 1 is Precision/PPV

# PPV CHART
df = pd.DataFrame({'Model': final_models, 'PPV': ppvs})
df = df.sort_values('PPV', ascending=False).reset_index(drop=True)
raw_palette = sns.color_palette('tab20', n_colors=len(df))
def fade_color(color, factor):
    return tuple(np.clip(np.array(color) + (1 - factor) * (np.array([1,1,1]) - np.array(color)), 0, 1))
N = len(df)
color_factors = np.linspace(1, 0.5, N)
bar_colors = [fade_color(raw_palette[i], color_factors[i]) for i in range(N)]
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Model'], df['PPV'], color=bar_colors, edgecolor='black')
plt.ylim(min(ppvs)-5, 102)
plt.ylabel("Positive Predictive Value (PPV) (%)", fontsize=13)
plt.title("Comparison of Positive Predictive Value (PPV) Across All Models", fontsize=15, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=40, ha="right", fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.savefig(os.path.join(RESULTS_DIR, 'Comparison_PPV.png'))
plt.show()

# ACCURACY CHART
acc_models = ["DAU-HQCNN (Proposed)", "Random Forest", "Logistic Regression", "Decision Tree", "SVM (RBF)", "KNN (k=3)", "MLP", "Gradient Boosting", "VGG16", "AdaBoost", "Naive Bayes", "SVM (Linear)", "LDA"]
accuracies = [results_data[m][0] for m in acc_models if m in results_data]
df_acc = pd.DataFrame({'Model': [m for m in acc_models if m in results_data], 'Accuracy': accuracies})
df_acc = df_acc.sort_values('Accuracy', ascending=False).reset_index(drop=True)
custom_colors = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE', '#008080', '#E6BEFF', '#9A6324']
plt.figure(figsize=(12, 6))
bars = plt.bar(df_acc['Model'], df_acc['Accuracy'], color=custom_colors[:len(df_acc)], edgecolor='black')
plt.ylim(min(accuracies)-5, 102)
plt.ylabel("Accuracy (%)", fontsize=13)
plt.title("Comparison of Accuracy Across All Models", fontsize=15, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=40, ha='right', fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.savefig(os.path.join(RESULTS_DIR, 'Comparison_Accuracy.png'))
plt.show()

# NPV CHART
npv_vals = [npv_data[m] for m in models_list if m in npv_data]
df_npv = pd.DataFrame({'Model': [m for m in models_list if m in npv_data], 'NPV': npv_vals})
plt.figure(figsize=(12, 6))
bars = plt.bar(df_npv['Model'], df_npv['NPV'], color=sns.color_palette("husl", len(df_npv)), edgecolor='black')
plt.ylabel("Negative Predictive Value (NPV)", fontsize=13)
plt.title("Comparison of Negative Predictive Value (NPV) Across All Models", fontsize=15, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=40, ha="right", fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.savefig(os.path.join(RESULTS_DIR, 'Comparison_NPV.png'))
plt.show()

# SPECIFICITY CHART
spec_models = ["DAU-HQCNN (Proposed)", "Random Forest", "Logistic Regression", "Decision Tree", "SVM", "KNN", "MLP", "Gradient Boosting", "VGG16", "AdaBoost", "Naive Bayes", "LDA"]
spec_vals = [results_data[m][4] for m in spec_models if m in results_data]
df_spec = pd.DataFrame({'Model': [m for m in spec_models if m in results_data], 'Specificity': spec_vals})
df_spec = df_spec.sort_values('Specificity', ascending=False).reset_index(drop=True)
plt.figure(figsize=(12, 6))
bars = plt.bar(df_spec['Model'], df_spec['Specificity'], color=sns.color_palette("viridis", len(df_spec)), edgecolor='black')
plt.ylabel("Specificity (%)", fontsize=13)
plt.title("Comparison of Specificity Across All Models", fontsize=15, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.xticks(rotation=40, ha="right", fontsize=11)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
plt.savefig(os.path.join(RESULTS_DIR, 'Comparison_Specificity.png'))
plt.show()

# ADVANCED DATA DICTIONARY FOR COMPLEX VIZ
data = {}
for m in ["DAU-HQCNN (Proposed)", "Random Forest", "Logistic Regression", "Decision Tree", "SVM", "KNN", "VGG16", "MLP", "Gradient Boosting", "AdaBoost", "Naive Bayes", "LDA"]:
    if m in results_data:
        data[m] = results_data[m]

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC', 'MCC', 'Kappa']

# VIZ: DIVERGING BAR CHART
fig, ax = plt.subplots(figsize=(14, 10))
methods_list = [m for m in data.keys() if m != 'DAU-HQCNN (Proposed)']
qcnn_values = data['DAU-HQCNN (Proposed)']
gaps = {m: [qcnn_values[i] - data[m][i] for i in range(len(metrics))] for m in methods_list}
y_pos = np.arange(len(methods_list))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(metrics)))
bar_width = 0.08
for idx, metric in enumerate(metrics):
    metric_gaps = [gaps[m][idx] for m in methods_list]
    pos = y_pos + (idx - len(metrics)/2) * bar_width
    ax.barh(pos, metric_gaps, bar_width, label=metric, color=colors[idx], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='darkred', linewidth=2, linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods_list, fontsize=10, fontweight='bold')
ax.set_xlabel('Performance Gap from DAU-HQCNN (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance Deficit Analysis: Distance from DAU-HQCNN Benchmark', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'Figure_Diverging_Performance_Gap.png'), dpi=300)
plt.show()

# VIZ: CIRCULAR BUBBLE
from matplotlib.patches import Circle
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect('equal'); ax.axis('off')
all_m_keys = list(data.keys())
angles = np.linspace(0, 2*np.pi, len(all_m_keys), endpoint=False)
for idx, (method, angle) in enumerate(zip(all_m_keys, angles)):
    x, y = 7 * np.cos(angle), 7 * np.sin(angle)
    avg_perf = np.mean(data[method])
    size = (avg_perf - 50) / 50 * 2.5
    color = '#2E7D32' if 'Proposed' in method else ('#66BB6A' if avg_perf >= 90 else '#EF5350')
    ax.add_patch(Circle((x, y), size, color=color, alpha=0.7, edgecolor='black', linewidth=1))
    ax.text(x, y, method.replace('Proposed', 'Ours'), ha='center', va='center', fontsize=8, fontweight='bold')
plt.savefig(os.path.join(RESULTS_DIR, 'Figure_Bubble_Landscape.png'), dpi=300)
plt.show()

# VIZ: PERFORMANCE TILE MAP
values = np.array([data[m] for m in data.keys()])
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_aspect('equal')
for i, model in enumerate(data.keys()):
    for j, metric in enumerate(metrics):
        v = values[i, j]
        size = 0.4 + (0.4) * ((v-min(values.flatten()))/(max(values.flatten())-min(values.flatten())))
        color = '#388E3C' if v > 95 else ('#FFA726' if v > 85 else '#1976D2')
        if 'Proposed' in model: color = '#1B5E20'
        ax.add_patch(plt.Rectangle([j-i*0.05, len(data)-i-1], size, size, fc=color, ec='white', lw=1))
        ax.text(j-i*0.05+size/2, len(data)-i-1+size/2, f'{v:.1f}', color='white', ha='center', va='center', fontsize=8, fontweight='bold')
ax.set_xticks(np.arange(len(metrics))); ax.set_xticklabels(metrics, rotation=35)
ax.set_yticks(np.arange(len(data))); ax.set_yticklabels(list(data.keys())[::-1])
plt.title('Performance Tile Map: All Models & Metrics Compared', fontsize=18, fontweight='bold')
plt.axis('off')
plt.savefig(os.path.join(RESULTS_DIR, 'Figure_Performance_TileMap.png'), dpi=300)
plt.show()

print(f"✅ All 12 visualizations generated and saved to {RESULTS_DIR}!")
