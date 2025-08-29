# dashboard.py - Final Copy-Paste Ready
# ---------------------------------------------
# Customer Insights Dashboard
# ---------------------------------------------

import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from utils import extract_and_load  # Ensure utils.py is in same folder

# -------------------------------
# Config
# -------------------------------
ZIP_PATH = r"C:\Users\himan\Downloads\archive (1).zip"
FOLDER_NAME = "Sales_Data"  # Only monthly CSVs

# -------------------------------
# Helper functions
# -------------------------------
def _metric_fmt_money(x, zero="-$"):
    if pd.isna(x):
        return zero
    return f"${x:,.0f}"

def _metric_fmt_money2(x, zero="$0.00"):
    if pd.isna(x):
        return zero
    return f"${x:,.2f}"

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def extract_region(addr):
    if pd.isna(addr) or addr is None:
        return None
    parts = str(addr).split(",")
    if len(parts) < 2:
        return None
    return parts[-2].strip()

def extract_city(addr):
    if pd.isna(addr) or addr is None:
        return None
    parts = str(addr).split(",")
    if len(parts) < 2:
        return None
    return parts[1].strip()

CITY_COORDS = {
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "Seattle": (47.6062, -122.3321),
    "Portland": (45.5152, -122.6784),
    "Austin": (30.2672, -97.7431),
    "Dallas": (32.7767, -96.7970),
    "Boston": (42.3601, -71.0589),
    "New York City": (40.7128, -74.0060),
    "Atlanta": (33.7490, -84.3880),
    "San Jose": (37.3382, -121.8863),
    "Denver": (39.7392, -104.9903),
    "Chicago": (41.8781, -87.6298),
}

def map_city_to_latlon(city):
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    return np.nan, np.nan

# -------------------------------
# Plotting helpers
# -------------------------------
def show_bar(series, title):
    st.markdown(f"### {title}")
    st.bar_chart(series)

def show_line(series, title):
    st.markdown(f"### {title}")
    st.line_chart(series)

# -------------------------------
# Dashboard main
# -------------------------------
def run_dashboard():
    st.title("📊 Customer Insights Dashboard")

    # Load data
    if not os.path.exists(ZIP_PATH):
        st.error(f"❌ Zip file not found: {ZIP_PATH}")
        st.stop()

    df, status = extract_and_load(ZIP_PATH, FOLDER_NAME)
    st.sidebar.info(status)

    if df.empty:
        st.error("⚠ No data loaded! Check ZIP and folder path.")
        st.stop()

    # Dataset overview
    st.subheader("Dataset Overview")
    st.write(df.head())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    # Ensure numeric
    df = ensure_numeric(df, ["Quantity Ordered", "Price Each"])

    # Compute sales if missing
    if "sales" not in df.columns and all(c in df.columns for c in ["Quantity Ordered","Price Each"]):
        df["sales"] = df["Quantity Ordered"] * df["Price Each"]

    # Extract Region/City
    if "Region" not in df.columns and "Purchase Address" in df.columns:
        df["Region"] = df["Purchase Address"].apply(extract_region)
    if "City" not in df.columns and "Purchase Address" in df.columns:
        df["City"] = df["Purchase Address"].apply(extract_city)

    # Date handling
    date_col = "Order Date" if "Order Date" in df.columns else "date"
    df[date_col] = safe_to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])
    df["year"] = df[date_col].dt.year.astype("Int64")
    df["month"] = df[date_col].dt.to_period("M")
    df["day_of_week"] = df[date_col].dt.day_name()
    df["hour"] = df[date_col].dt.hour

    # Sidebar filters
    st.sidebar.header("Filters")
    years_available = sorted(df["year"].dropna().unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", years_available, default=years_available)
    regions_available = sorted(df["Region"].dropna().unique()) if "Region" in df.columns else []
    selected_regions = st.sidebar.multiselect("Select Region(s)", regions_available, default=regions_available)

    df_filtered = df.copy()
    if selected_years:
        df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]
    if selected_regions and "Region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Region"].isin(selected_regions)]
    if df_filtered.empty:
        st.warning("⚠ No data after filters.")
        st.stop()

    # KPIs
    st.subheader("Key Metrics")
    total_sales = float(df_filtered["sales"].sum())
    avg_sales = float(df_filtered["sales"].mean())
    max_sales = float(df_filtered["sales"].max())
    num_orders = int(df_filtered["Order ID"].nunique()) if "Order ID" in df_filtered.columns else int(df_filtered.shape[0])
    num_products = int(df_filtered["Product"].nunique()) if "Product" in df_filtered.columns else 0

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("💰 Total Sales", _metric_fmt_money(total_sales))
    k2.metric("📊 Average Sale/Row", _metric_fmt_money(avg_sales))
    k3.metric("🔥 Max Sale", _metric_fmt_money(max_sales))
    k4.metric("🧾 Orders (proxy)", f"{num_orders:,}")

    # Histograms
    numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col_x = st.selectbox("Select column for histogram", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df_filtered[col_x], kde=True, ax=ax)
        st.pyplot(fig)

    # Line charts
    if "month" in df_filtered.columns:
        monthly_sales = df_filtered.groupby("month")["sales"].sum().reset_index().sort_values("month")
        show_line(monthly_sales.set_index("month")["sales"], "Monthly Sales Trend")

    # Region Sales
    if "Region" in df_filtered.columns:
        region_sales = df_filtered.groupby("Region")["sales"].sum().sort_values(ascending=False)
        show_bar(region_sales, "Sales by Region")

    # City Sales & Map
    if "City" in df_filtered.columns:
        city_sales = df_filtered.groupby("City")["sales"].sum().reset_index().sort_values("sales", ascending=False)
        st.subheader("Top Cities by Sales")
        st.dataframe(city_sales.head(10), use_container_width=True)
        city_sales_map = city_sales.copy()
        city_sales_map[["lat","lon"]] = city_sales_map["City"].apply(lambda c: pd.Series(map_city_to_latlon(c)))
        city_sales_map = city_sales_map.dropna(subset=["lat","lon"])
        if not city_sales_map.empty:
            st.map(city_sales_map.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]])

    # Top Products
    if "Product" in df_filtered.columns:
        top_products = df_filtered.groupby("Product")["sales"].sum().sort_values(ascending=False).head(10)
        show_bar(top_products, "Top 10 Products by Sales")

    # Profit
    if all(c in df_filtered.columns for c in ["Price Each", "Quantity Ordered"]):
        df_filtered["Cost Price"] = df_filtered["Price Each"]*0.7
        df_filtered["Profit"] = (df_filtered["Price Each"]-df_filtered["Cost Price"])*df_filtered["Quantity Ordered"]
        st.subheader("Profit Analysis")
        total_profit = df_filtered["Profit"].sum()
        avg_profit = df_filtered["Profit"].mean()
        profit_margin = total_profit/total_sales*100 if total_sales else 0
        p1,p2,p3 = st.columns(3)
        p1.metric("💰 Total Profit", _metric_fmt_money(total_profit))
        p2.metric("🧾 Avg Profit per Row", _metric_fmt_money(avg_profit))
        p3.metric("📈 Profit Margin", f"{profit_margin:.2f}%")

    # Prophet Forecasting
    if "sales" in df_filtered.columns:
        st.subheader("Forecasting (Prophet)")
        try:
            monthly = df_filtered.resample("M", on=date_col)["sales"].sum().reset_index()
            monthly.columns = ["ds","y"]
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(monthly)
            future = m.make_future_dataframe(periods=6, freq="M")
            forecast = m.predict(future)
            fig1 = m.plot(forecast)
            st.pyplot(fig1)
        except Exception as e:
            st.warning(f"⚠ Prophet failed: {e}")
    else:
        st.info("Prophet forecasting skipped: 'sales' column missing.")

    # Excel Download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_filtered.to_excel(writer, index=False, sheet_name="FilteredData")
    st.download_button("⬇ Download Filtered Data (Excel)", excel_buffer.getvalue(), file_name="sales_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    run_dashboard()
