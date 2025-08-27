from utils import extract_and_load
import os

# Sirf File 1 test ke liye
df1, _ = extract_and_load(
    r"C:\Users\himan\Downloads\archive (1).zip",
    os.path.join("extracted_files", "file1")
)

print("\n🔹 Columns in File 1 (Sales Data):")
print(df1.columns.tolist())
