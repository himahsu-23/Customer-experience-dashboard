from utils import extract_and_load
import os
import sys

# Check if running on Streamlit Cloud
if "STREAMLIT_SERVER" in os.environ:
    zip_path = "archive.zip"  # file in repo
else:
    zip_path = r"C:\Users\himan\Downloads\archive (1).zip"  # local path

extract_path = os.path.join("extracted_files", "file1")
df1, _ = extract_and_load(zip_path, extract_path)

print("\n🔹 Columns in File 1 (Sales Data):")
print(df1.columns.tolist())
