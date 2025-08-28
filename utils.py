import pandas as pd
import zipfile
import io

def extract_and_load(zip_path, folder_name):
    """
    Extracts and loads multiple CSV files from a zip archive into a single DataFrame.
    
    Args:
        zip_path (str): Path to the zip file.
        folder_name (str): The folder inside the zip where CSVs are located.
    
    Returns:
        pd.DataFrame: A single DataFrame with all data concatenated.
    """
    all_dfs = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_name in z.namelist():
                # Check if the file is a CSV and is in the specified folder
                if file_name.startswith(folder_name) and file_name.endswith('.csv'):
                    with z.open(file_name) as f:
                        # Use io.BytesIO to read the file content as a stream
                        # This avoids writing temporary files to disk
                        df = pd.read_csv(io.BytesIO(f.read()))
                        all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df, "Data loaded successfully."
        else:
            return pd.DataFrame(), "No CSV files found in the specified folder."
            
    except FileNotFoundError:
        return pd.DataFrame(), "Error: Zip file not found."
    except Exception as e:
        return pd.DataFrame(), f"An error occurred: {e}"