# utils.py
import os
import zipfile
import io
import pandas as pd

def extract_and_load(zip_path: str, folder_name: str = "", read_csv_kwargs: dict = None):
    """
    Extract CSV files from zip and load into one DataFrame.
    - zip_path: full path to the zip archive
    - folder_name: subfolder inside the zip to filter (e.g. "Sales_Data"); if empty => load all CSVs
    - read_csv_kwargs: optional dict forwarded to pd.read_csv
    Returns:
        (pd.DataFrame, str)  -> combined dataframe and status message
    """
    read_csv_kwargs = read_csv_kwargs or {}
    if not os.path.exists(zip_path):
        return pd.DataFrame(), f"❌ Zip not found: {zip_path}"

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            all_names = z.namelist()

            # Normalize folder_name: allow both "Sales_Data" and "Sales_Data/" matches
            def matches_target(n):
                if not n.lower().endswith(".csv"):
                    return False
                if folder_name:
                    # accept if path segments startwith folder_name
                    return n.startswith(folder_name) or n.startswith(folder_name + "/") or (("/" + folder_name + "/") in n)
                return True

            csv_files = [n for n in all_names if matches_target(n)]

            if not csv_files:
                # helpful debug message: list top-level entries
                sample = ", ".join(all_names[:10]) + ("..." if len(all_names) > 10 else "")
                return pd.DataFrame(), f"⚠️ No CSV files found in '{folder_name or 'ZIP root'}'. ZIP contents sample: {sample}"

            dfs = []
            for f in csv_files:
                try:
                    with z.open(f) as fh:
                        # try utf-8 first, then fallback to latin1 if needed
                        try:
                            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), **read_csv_kwargs)
                        except Exception:
                            fh.seek(0)
                            df = pd.read_csv(io.TextIOWrapper(fh, encoding="latin1"), **read_csv_kwargs)
                        dfs.append(df)
                except Exception as e:
                    return pd.DataFrame(), f"❌ Failed reading {f}: {e}"

        # concat safely
        if not dfs:
            return pd.DataFrame(), "⚠️ No CSVs were read successfully."
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        return combined, f"✅ Loaded {len(csv_files)} CSV file(s) from {folder_name or 'ZIP root'}."
    except zipfile.BadZipFile:
        return pd.DataFrame(), "❌ The file is not a valid ZIP archive."
    except Exception as e:
        return pd.DataFrame(), f"❌ Error extracting/reading ZIP: {e}"
