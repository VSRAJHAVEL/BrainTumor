import os
import glob
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm

def normalize_slice(img_slice):
    # Normalize image slice to 0-255
    if np.max(img_slice) == 0:
        return np.zeros_like(img_slice, dtype=np.uint8)
    
    img_norm = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
    img_norm = (img_norm * 255).astype(np.uint8)
    return img_norm

def preprocess_brats():
    base_dir = r"D:\Brain\Datasets\BraTS\Task01_BrainTumour"
    img_dir = os.path.join(base_dir, "imagesTr")
    lbl_dir = os.path.join(base_dir, "labelsTr")
    
    # Destination directories for Classification and Segmentation
    dest_cls_yes = r"D:\Brain\Datasets\BraTS\classification\yes"
    dest_cls_no = r"D:\Brain\Datasets\BraTS\classification\no"
    
    dest_seg_img = r"D:\Brain\Datasets\BraTS\segmentation\images"
    dest_seg_mask = r"D:\Brain\Datasets\BraTS\segmentation\masks"
    
    os.makedirs(dest_cls_yes, exist_ok=True)
    os.makedirs(dest_cls_no, exist_ok=True)
    os.makedirs(dest_seg_img, exist_ok=True)
    os.makedirs(dest_seg_mask, exist_ok=True)
    
    # Find all NIfTI files
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    
    if not img_files:
        print(f"No NIfTI files found in {img_dir}. Waiting for download/extraction?")
        return
        
    print(f"Found {len(img_files)} NIfTI files. Processing...")
    
    # Process only a subset or all, taking a few slices per volume to avoid massive dataset
    for img_path in tqdm(img_files):
        filename = os.path.basename(img_path).replace('.nii.gz', '')
        lbl_path = os.path.join(lbl_dir, os.path.basename(img_path))
        
        if not os.path.exists(lbl_path):
            continue
            
        try:
            # Load images
            img_obj = nib.load(img_path)
            lbl_obj = nib.load(lbl_path)
            
            img_data = img_obj.get_fdata() # shape: (H, W, D, 4 modalities)
            lbl_data = lbl_obj.get_fdata() # shape: (H, W, D)
            
            # Using FLAIR modality (index 0)
            flair_data = img_data[:, :, :, 0]
            
            num_slices = flair_data.shape[2]
            
            # Find slices with tumor
            tumor_slices = []
            no_tumor_slices = []
            
            for i in range(num_slices):
                slice_lbl = lbl_data[:, :, i]
                if np.sum(slice_lbl > 0) > 100: # Has a reasonable size tumor
                    tumor_slices.append(i)
                elif np.sum(flair_data[:, :, i] > 0) > 1000: # Has brain tissue but no tumor
                    no_tumor_slices.append(i)
            
            # Select up to 5 tumor slices and 2 no-tumor slices per volume
            selected_tumor = tumor_slices[len(tumor_slices)//2 - 2 : len(tumor_slices)//2 + 3] if len(tumor_slices) >= 5 else tumor_slices
            selected_notumor = no_tumor_slices[len(no_tumor_slices)//2 - 1 : len(no_tumor_slices)//2 + 1] if len(no_tumor_slices) >= 2 else no_tumor_slices
            
            # Process YES (Tumor)
            for idx, slice_idx in enumerate(selected_tumor):
                slice_img = flair_data[:, :, slice_idx]
                slice_lbl = lbl_data[:, :, slice_idx]
                
                # Normalize image
                img_norm = normalize_slice(slice_img)
                
                # Binarize mask
                mask_bin = (slice_lbl > 0).astype(np.uint8) * 255
                
                # Rotate 90 degrees to make it upright (BraTS slices are usually rotated)
                img_norm = np.rot90(img_norm)
                mask_bin = np.rot90(mask_bin)
                
                out_name = f"{filename}_slice{slice_idx}.png"
                
                # Save to classification 'yes'
                cv2.imwrite(os.path.join(dest_cls_yes, out_name), img_norm)
                
                # Save to segmentation
                cv2.imwrite(os.path.join(dest_seg_img, out_name), img_norm)
                cv2.imwrite(os.path.join(dest_seg_mask, out_name), mask_bin)
                
            # Process NO (No Tumor)
            for idx, slice_idx in enumerate(selected_notumor):
                slice_img = flair_data[:, :, slice_idx]
                
                # Normalize image
                img_norm = normalize_slice(slice_img)
                img_norm = np.rot90(img_norm)
                
                out_name = f"{filename}_slice{slice_idx}.png"
                
                # Save to classification 'no'
                cv2.imwrite(os.path.join(dest_cls_no, out_name), img_norm)
                
                # Note: Not saving to segmentation since masks are all empty, unless we want to evaluate TN
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("Finished preprocessing BraTS dataset.")

if __name__ == "__main__":
    preprocess_brats()
