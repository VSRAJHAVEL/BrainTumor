import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    
    with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    base_dir = r"D:\Brain\Datasets\BraTS"
    os.makedirs(base_dir, exist_ok=True)
    
    # URL for Medical Segmentation Decathlon Task01_BrainTumour
    tar_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    tar_path = os.path.join(base_dir, "Task01_BrainTumour.tar")
    
    if not os.path.exists(tar_path):
        download_file(tar_url, tar_path)
    else:
        print(f"Archive {tar_path} already exists. Skipping download.")
        
    print("Extracting tar archive... (This will take a while)")
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(base_dir)
    
    print("Done downloading and extracting BraTS dataset.")

if __name__ == "__main__":
    main()
