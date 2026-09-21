import os
import glob
import h5py
import numpy as np
import cv2
from tqdm import tqdm

def preprocess_figshare():
    src_dir = r"D:\Brain\Datasets\Figshare"
    
    # Destination directories for Classification (yes/no) and Segmentation
    dest_cls_yes = r"D:\Brain\Datasets\Figshare\classification\yes"
    dest_cls_no = r"D:\Brain\Datasets\Figshare\classification\no"
    
    dest_seg_img = r"D:\Brain\Datasets\Figshare\segmentation\images"
    dest_seg_mask = r"D:\Brain\Datasets\Figshare\segmentation\masks"
    
    os.makedirs(dest_cls_yes, exist_ok=True)
    os.makedirs(dest_cls_no, exist_ok=True)
    os.makedirs(dest_seg_img, exist_ok=True)
    os.makedirs(dest_seg_mask, exist_ok=True)
    
    mat_files = glob.glob(os.path.join(src_dir, "*.mat"))
    print(f"Found {len(mat_files)} .mat files. Processing...")
    
    for mat_path in tqdm(mat_files):
        filename = os.path.basename(mat_path).replace('.mat', '')
        if filename == 'cvind': # skip cross validation index file
            continue
            
        try:
            with h5py.File(mat_path, 'r') as f:
                cjdata = f['cjdata']
                image = np.array(cjdata['image'])
                mask = np.array(cjdata['tumorMask'])
                label = np.array(cjdata['label'])[0][0] # 1: meningioma, 2: glioma, 3: pituitary
            
            # Normalize image for saving as PNG
            image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Mask is already 0/1, convert to 0/255
            mask_norm = (mask * 255).astype(np.uint8)
            
            # All Figshare data contains tumors, so we put them in 'yes' for classification
            cls_out_path = os.path.join(dest_cls_yes, f"{filename}.png")
            cv2.imwrite(cls_out_path, image_norm)
            
            # Save for segmentation
            seg_img_out = os.path.join(dest_seg_img, f"{filename}.png")
            seg_mask_out = os.path.join(dest_seg_mask, f"{filename}.png")
            
            cv2.imwrite(seg_img_out, image_norm)
            cv2.imwrite(seg_mask_out, mask_norm)
            
        except Exception as e:
            print(f"Error processing {mat_path}: {e}")

    print("Finished preprocessing Figshare dataset.")

if __name__ == "__main__":
    preprocess_figshare()
