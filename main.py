# main.py
import streamlit as st
from utils import extract_and_load
from dashboard import run_dashboard

st.set_page_config(page_title="📊 Customer Insights Dashboard", layout="wide")

ZIP_PATH = r"C:\Users\himan\Downloads\archive (1).zip"
FOLDER_NAME = ""  # load all CSVs (Sales_Data + Updated_sales.csv). Use "Sales_Data" to load only that folder.

df, status = extract_and_load(ZIP_PATH, FOLDER_NAME)
st.sidebar.info(status)

if df is None or df.empty:
    st.error("❌ No data loaded. Check ZIP path / folder name.")
else:
    run_dashboard(df)
