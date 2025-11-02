import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import zipfile
import shutil # For robust directory removal

BASE_RAW_DATA_PATH = "data/raw"

def download_ett_data():
    """
    Downloads ETT datasets from the ETDataset GitHub repository.
    """
    raw_data_path = os.path.join(BASE_RAW_DATA_PATH, "ETT")
    print(f"Downloading ETT data to {raw_data_path}...")
    os.makedirs(raw_data_path, exist_ok=True)

    ett_base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
    ett_files = {
        "ETTh1.csv": ett_base_url + "ETTh1.csv",
        "ETTh2.csv": ett_base_url + "ETTh2.csv",
        "ETTm1.csv": ett_base_url + "ETTm1.csv",
        "ETTm2.csv": ett_base_url + "ETTm2.csv",
    }

    for filename, url in ett_files.items():
        file_path = os.path.join(raw_data_path, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename} from {url}...")
            try:
                os.system(f"wget -q -O {file_path} {url}")
                print(f"Successfully downloaded {filename}.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists. Skipping download.")
    print("ETT data download complete.")

def download_pems_data():
    """
    Downloads PEMS datasets from the juyongjiang/TimeSeriesDatasets GitHub repository.
    """
    raw_data_path = os.path.join(BASE_RAW_DATA_PATH, "PEMS")
    print(f"Downloading PEMS data to {raw_data_path}...")
    os.makedirs(raw_data_path, exist_ok=True)

    pems_base_url = "https://raw.githubusercontent.com/juyongjiang/TimeSeriesDatasets/main/"
    pems_files = {
        "PEMS03.csv": pems_base_url + "PEMS03.csv",
        "PEMS04.csv": pems_base_url + "PEMS04.csv",
        "PEMS07.csv": pems_base_url + "PEMS07.csv",
        "PEMS08.csv": pems_base_url + "PEMS08.csv",
    }

    for filename, url in pems_files.items():
        file_path = os.path.join(raw_data_path, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename} from {url}...")
            try:
                os.system(f"wget -q -O {file_path} {url}")
                print(f"Successfully downloaded {filename}.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists. Skipping download.")
    print("PEMS data download complete.")

def download_electricity_data():
    """
    Downloads and extracts the Electricity dataset from UCI Machine Learning Repository.
    """
    raw_data_path = os.path.join(BASE_RAW_DATA_PATH, "Electricity")
    print(f"Downloading Electricity data to {raw_data_path}...")
    os.makedirs(raw_data_path, exist_ok=True)

    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
    zip_filename = os.path.join(raw_data_path, "LD2011_2014.txt.zip")
    extracted_txt_filename = "LD2011_2014.txt"
    final_filename = os.path.join(raw_data_path, "Electricity.csv")
    temp_extract_dir = os.path.join(raw_data_path, "temp_electricity_extract")

    if os.path.exists(final_filename):
        print(f"{final_filename} already exists. Skipping download and extraction.")
        print("Electricity data download complete.")
        return

    # Clean up previous attempts if any
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)

    print(f"Downloading {zip_filename.split(os.sep)[-1]} from {zip_url} using requests...")
    try:
        response = requests.get(zip_url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {zip_filename.split(os.sep)[-1]}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {zip_filename.split(os.sep)[-1]}: {e}")
        return

    print(f"Extracting {zip_filename.split(os.sep)[-1]}...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        extracted_file_path = os.path.join(temp_extract_dir, extracted_txt_filename)
        if os.path.exists(extracted_file_path):
            os.rename(extracted_file_path, final_filename)
            print(f"Successfully extracted and moved {extracted_txt_filename} to {final_filename}.")
        else:
            print(f"Error: Expected file {extracted_txt_filename} not found after extraction.")

    except zipfile.BadZipFile:
        print(f"Error: {zip_filename} is a bad zip file. It might be corrupted.")
    except Exception as e:
        print(f"Error during extraction or moving Electricity data: {e}")
    finally:
        # Clean up
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
    print("Electricity data download complete.")

def download_weather_data():
    """
    Placeholder for Weather dataset download.
    The exact dataset used in the paper is not directly specified or easily found.
    Users might need to manually obtain a suitable Weather dataset or use a common alternative.
    """
    raw_data_path = os.path.join(BASE_RAW_DATA_PATH, "Weather")
    print("\n--- Weather Dataset ---")
    print("The exact Weather dataset used in the paper is not directly specified or easily found.")
    print("Please consider manually obtaining a suitable Weather dataset (e.g., from UCI, Kaggle, or a processed version from other research codebases).")
    print(f"Place the Weather dataset CSV file (e.g., 'Weather.csv') in: {raw_data_path}")
    print("-----------------------")

def download_solar_data():
    """
    Placeholder for Solar-Energy dataset download.
    The exact dataset used in the paper is not directly specified or easily found.
    Users might need to manually obtain a suitable Solar-Energy dataset or use a common alternative.
    """
    raw_data_path = os.path.join(BASE_RAW_DATA_PATH, "Solar")
    print("\n--- Solar-Energy Dataset ---")
    print("The exact Solar-Energy dataset used in the paper is not directly specified or easily found.")
    print("Please consider manually obtaining a suitable Solar-Energy dataset (e.g., from NREL, Kaggle, or a processed version from other research codebases).")
    print(f"Place the Solar-Energy dataset CSV file (e.g., 'Solar.csv') in: {raw_data_path}")
    print("--------------------------")

def download_all_datasets():
    download_ett_data()
    download_pems_data()
    download_electricity_data()
    download_weather_data()
    download_solar_data()

if __name__ == "__main__":
    download_all_datasets()