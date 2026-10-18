import os
import urllib.request
import zipfile

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded: {destination} ({os.path.getsize(destination)} bytes)")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "jena_climate_2009_2016.csv")
    if os.path.exists(csv_path):
        print(f"Dataset already exists at: {csv_path}")
        return

    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = os.path.join(data_dir, "jena_climate_2009_2016.csv.zip")
    
    print("\n==========================================")
    print("Downloading Jena Climate Dataset (MPI-BGC)")
    print("==========================================")
    if download_file(url, zip_path):
        print(f"Extracting zip archive: {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print(f"Extracted to {csv_path} and cleaned up zip archive.")

    print("\nDataset preparation completed!")

if __name__ == "__main__":
    main()
