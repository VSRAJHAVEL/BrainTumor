import os
import requests
import zipfile
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
    base_dir = r"D:\Brain\Datasets\Figshare"
    os.makedirs(base_dir, exist_ok=True)
    
    # Download the main zip file
    main_zip_url = "https://ndownloader.figshare.com/articles/1512427/versions/5"
    main_zip_path = os.path.join(base_dir, "figshare_brain_tumor.zip")
    
    if not os.path.exists(main_zip_path):
        download_file(main_zip_url, main_zip_path)
    else:
        print(f"Main zip {main_zip_path} already exists. Skipping download.")
        
    print("Extracting main zip file...")
    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
        
    # Extract nested zip files
    for file in os.listdir(base_dir):
        if file.endswith(".zip") and file != "figshare_brain_tumor.zip":
            zip_path = os.path.join(base_dir, file)
            print(f"Extracting nested zip {file}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
    
    print("Done downloading and extracting Figshare dataset.")

if __name__ == "__main__":
    main()
