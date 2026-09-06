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
    
    datasets = [
        {
            "name": "Jena Climate Dataset (MPI-BGC)",
            "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            "filename": "jena_climate_2009_2016.csv.zip",
            "is_zip": True
        },
        {
            "name": "Numenta Anomaly Benchmark (NAB) - Ambient Temperature System Failure",
            "url": "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv",
            "filename": "nab_ambient_temperature_system_failure.csv",
            "is_zip": False
        },
        {
            "name": "Chicago Beach Weather Stations Automated Sensors",
            "url": "https://data.cityofchicago.org/api/views/k7hf-8y75/rows.csv?accessType=DOWNLOAD",
            "filename": "chicago_beach_weather_automated_sensors.csv",
            "is_zip": False
        }
    ]

    for ds in datasets:
        dest_path = os.path.join(data_dir, ds["filename"])
        print(f"\n==========================================")
        print(f"Processing: {ds['name']}")
        print(f"==========================================")
        if download_file(ds["url"], dest_path):
            if ds.get("is_zip"):
                print(f"Extracting zip archive: {dest_path}...")
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Successfully extracted zip contents to {data_dir}")

    print("\nDataset download process completed!")

if __name__ == "__main__":
    main()
