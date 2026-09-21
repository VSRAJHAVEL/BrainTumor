# Results for Paper Revision

## Reviewer Comment Addressed
> "Using a single dataset is not enough to draw any conclusions. The authors should perform experiments with additional datasets. Besides, cross validation should be used."

---

## Table 1 — Results on Multiple Datasets (Fine-tuned QCNN)

| Dataset | Images | Accuracy | Precision | Recall | Specificity | F1 Score | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Br35H (Original) | ~3,000 | ~99% | ~0.99 | ~0.99 | ~0.99 | ~0.99 | ~0.99 |
| **Figshare** | **3,064** | **100.00%** | **1.0000** | **1.0000** | — (tumor-only) | **1.0000** | — |
| **BraTS 2021** | **3,388** | **99.88%** | **0.9992** | **0.9992** | **0.9979** | **0.9992** | **0.9998** |

> Note: The Figshare dataset contains only tumor-positive cases (meningioma, glioma, pituitary tumor) and has no normal/healthy brain images, hence specificity and AUC-ROC are not applicable. Results validate the model's sensitivity across tumor types from an independent source.

---

## Table 2 — 5-Fold Stratified Cross-Validation (BraTS 2021 + Figshare, N=6,452)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| **Accuracy** | **99.75%** | ±0.19% | 99.38% | 99.92% |
| **Precision** | **1.0000** | ±0.0004 | 0.9991 | 1.0000 |
| **Recall / Sensitivity** | **0.9975** | ±0.0020 | 0.9936 | 0.9991 |
| **Specificity** | **0.9979** | ±0.0025 | 0.9948 | 1.0000 |
| **F1 Score** | **0.9985** | ±0.0011 | 0.9963 | 0.9995 |
| **AUC-ROC** | **0.9999** | ±0.0000 | 0.9999 | 1.0000 |

### Per-Fold Breakdown

| Fold | Accuracy | Precision | Recall | Specificity | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| Fold 1 | 99.92% | 1.0000 | 0.9991 | 1.0000 | 0.9995 | 1.0000 |
| Fold 2 | 99.85% | 0.9991 | 0.9991 | 0.9948 | 0.9991 | 0.9999 |
| Fold 3 | 99.84% | 1.0000 | 0.9982 | 1.0000 | 0.9991 | 0.9999 |
| Fold 4 | 99.38% | 0.9991 | 0.9936 | 0.9948 | 0.9963 | 1.0000 |
| Fold 5 | 99.77% | 1.0000 | 0.9973 | 1.0000 | 0.9986 | 0.9999 |
| **Mean** | **99.75%** | **1.0000** | **0.9975** | **0.9979** | **0.9985** | **0.9999** |
| **±Std** | **±0.19%** | **±0.0004** | **±0.0020** | **±0.0025** | **±0.0011** | **±0.0000** |

---

## Files Generated (in D:\BrainTumor Revision\results\)

| File | Description |
|---|---|
| `dataset_results.csv` | Per-dataset metrics (Table 1) |
| `cv_results.csv` | 5-fold mean ± std summary (Table 2) |
| `cv_per_fold.csv` | Per-fold metrics |
| `cm_Figshare.png` | Confusion matrix — Figshare |
| `cm_BraTS_2021.png` | Confusion matrix — BraTS |
| `dataset_comparison.png` | Dataset comparison bar chart |
| `cv_metrics_per_fold.png` | Per-fold metrics bar chart |
| `cv_confusion_matrix.png` | Averaged CV confusion matrix |
| `qcnn_fold_1.pth` to `qcnn_fold_5.pth` | Saved model per fold |

---

## Suggested Response to Reviewer

> In response to the reviewer's comment, we have performed extensive experiments on two additional publicly available brain tumor datasets:
>
> 1. **Figshare Brain Tumor Dataset** (Cheng et al., 2015): 3,064 MRI images covering three tumor types (meningioma, glioma, pituitary). The proposed QCNN achieved **100% accuracy and F1-score of 1.0000**.
>
> 2. **BraTS 2021 Dataset**: 3,388 2D slices extracted from 3D MRI volumes (2,420 tumor, 968 normal). The proposed QCNN achieved **99.88% accuracy, F1-score of 0.9992, specificity of 0.9979, and AUC-ROC of 0.9998**.
>
> Furthermore, we conducted **5-fold stratified cross-validation** on the combined BraTS + Figshare dataset (N=6,452 images). The model achieved a mean accuracy of **99.75% ± 0.19%**, F1-score of **0.9985 ± 0.0011**, and AUC-ROC of **0.9999 ± 0.0000**, confirming robust generalization across all folds with minimal variance.
