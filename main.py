import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import extract_and_load # This now refers to the new function

st.set_page_config(page_title="📊 Customer Insights Dashboard", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
# Now, `extract_and_load` will get multiple years' data
df1, _ = extract_and_load(
    r"C:\Users\himan\Downloads\archive (1).zip",
    os.path.join("extracted_files", "file1")
)

# -------------------------------
# Data Cleaning & Feature Engineering
# -------------------------------
# ... (the rest of your script remains the same)