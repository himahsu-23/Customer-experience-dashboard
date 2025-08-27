import zipfile
import os
import pandas as pd

def extract_and_load(zip_path, extract_dir):
    """
    Zip extract + CSV load karne ka helper
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Extracted folder me CSV dhoondo
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found in extracted folder")

    csv_path = os.path.join(extract_dir, csv_files[0])
    df = pd.read_csv(csv_path, encoding="latin1")

    # Cleaning
    df.dropna(how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    return df, csv_path
