"""
Preprocess TCGA-LGG dataset for binary classification.
Uses tumor segmentation masks to label each slice:
  - Mask has non-zero pixels -> tumor (yes)
  - Mask is all zeros        -> normal (no)

Dataset: mateuszbuda/lgg-mri-segmentation (Kaggle)
Source : The Cancer Imaging Archive (TCIA) / NIH
"""
import os, glob, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_DIR = r"D:\Brain\Datasets\TCGA_LGG"
OUT_YES = r"D:\Brain\Datasets\TCGA_LGG\classification\yes"
OUT_NO  = r"D:\Brain\Datasets\TCGA_LGG\classification\no"

os.makedirs(OUT_YES, exist_ok=True)
os.makedirs(OUT_NO,  exist_ok=True)

# The dataset structure:
# lgg-mri-segmentation/kaggle_3m/<patient_id>/<patient_id>_<N>.tif     (MRI slice)
#                                             <patient_id>_<N>_mask.tif  (mask)

# Find all mask files
mask_files = glob.glob(os.path.join(RAW_DIR, "**", "*_mask.tif"), recursive=True)
print(f"Found {len(mask_files)} mask files")

tumor_count  = 0
normal_count = 0
skipped      = 0

for mask_path in tqdm(mask_files, desc="Processing"):
    # Derive the corresponding MRI image path
    img_path = mask_path.replace("_mask.tif", ".tif")
    if not os.path.exists(img_path):
        skipped += 1
        continue

    # Read mask - if any non-zero pixel exists -> tumor slice
    mask = np.array(Image.open(mask_path).convert("L"))
    has_tumor = mask.max() > 0

    # Copy MRI image (not mask) to the appropriate folder
    fname = os.path.basename(img_path)
    if has_tumor:
        dest = os.path.join(OUT_YES, fname)
        tumor_count += 1
    else:
        dest = os.path.join(OUT_NO, fname)
        normal_count += 1

    if not os.path.exists(dest):
        shutil.copy2(img_path, dest)

print("\n" + "="*50)
print("TCGA-LGG Preprocessing Complete")
print("="*50)
print(f"  Tumor  (yes): {tumor_count}")
print(f"  Normal (no) : {normal_count}")
print(f"  Skipped     : {skipped}")
print(f"  Total       : {tumor_count + normal_count}")
print(f"\n  Output -> {os.path.dirname(OUT_YES)}")
