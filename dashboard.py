import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from utils import extract_and_load

# PDF export
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Prophet forecasting
from prophet import Prophet

# -------------------------------
# Page config
st.set_page_config(page_title="📊 Customer Insights Dashboard", layout="wide")

# -------------------------------
# File Upload / Load Data
st.sidebar.header("🔍 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload ZIP file (Sales Data)", type="zip")

if uploaded_file:
    zip_path = f"temp_{uploaded_file.name}"
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df1, _ = extract_and_load(zip_path, os.path.join("extracted_files", "file1"))

elif os.path.exists("archive.zip"):
    df1, _ = extract_and_load("archive.zip", os.path.join("extracted_files", "file1"))

else:
    st.error("⚠️ Please upload a ZIP file containing your CSV data!")
    st.stop()

# -------------------------------
# Data Cleaning & Feature Engineering
df1.dropna(inplace=True)
df1["Order Date"] = pd.to_datetime(df1["Order Date"], errors="coerce")
df1 = df1.dropna(subset=["Order Date"])
df1["year"] = df1["Order Date"].dt.year
df1["Quantity Ordered"] = pd.to_numeric(df1["Quantity Ordered"], errors="coerce")
df1["Price Each"] = pd.to_numeric(df1["Price Each"], errors="coerce")
df1["sales"] = df1["Quantity Ordered"] * df1["Price Each"]
df1["Region"] = df1["Purchase Address"].apply(
    lambda x: str(x).split(",")[-2].strip() if pd.notnull(x) else None
)

# -------------------------------
# Sidebar Filters
year = st.sidebar.selectbox("Select Year", sorted(df1["year"].unique()))
region = st.sidebar.multiselect("Select Region", df1["Region"].unique())
df_filtered = df1[df1["year"] == year]
if region:
    df_filtered = df_filtered[df_filtered["Region"].isin(region)]

# -------------------------------
# Dashboard Title
st.title("📈 Customer Insights Dashboard")
st.write("Analyze sales trends by year and region.")

# -------------------------------
# Key Metrics
total_sales = df_filtered["sales"].sum()
avg_sales = df_filtered["sales"].mean()
max_sales = df_filtered["sales"].max()
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📊 Average Sales", f"${avg_sales:,.0f}")
col3.metric("🔥 Highest Sale", f"${max_sales:,.0f}")

# -------------------------------
# Sales by Region
if not df_filtered.empty:
    region_sales = df_filtered.groupby("Region")["sales"].sum().sort_values(ascending=False)
    st.subheader("📊 Sales by Region")
    st.bar_chart(region_sales)

# -------------------------------
# Yearly Sales Trend
yearly_sales = df1.groupby("year")["sales"].sum()
st.subheader("📈 Yearly Sales Trend")
st.line_chart(yearly_sales)

# -------------------------------
# Monthly Sales Trend
df1["month"] = df1["Order Date"].dt.to_period("M")
monthly_sales = df1.groupby("month")["sales"].sum().reset_index()
monthly_sales["month"] = monthly_sales["month"].astype(str)
st.subheader("📅 Monthly Sales Trend")
st.line_chart(monthly_sales.set_index("month"))

# -------------------------------
# Day-of-Week Sales
df1["day_of_week"] = df1["Order Date"].dt.day_name()
day_sales = df1.groupby("day_of_week")["sales"].sum().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
st.subheader("📆 Sales by Day of Week")
st.bar_chart(day_sales)

# -------------------------------
# Top Products
st.subheader("📦 Top 10 Products by Quantity Sold")
if "Product" in df1.columns:
    top_products_qty = df1.groupby("Product")["Quantity Ordered"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products_qty)
else:
    st.warning("⚠️ 'Product' column missing in data")

# -------------------------------
# Profit Analysis
df1["Cost Price"] = df1["Price Each"] * 0.7
df1["Profit"] = (df1["Price Each"] - df1["Cost Price"]) * df1["Quantity Ordered"]

total_profit = df1["Profit"].sum()
avg_profit = df1["Profit"].mean()
profit_margin = (df1["Profit"].sum() / df1["sales"].sum()) * 100

p1, p2, p3 = st.columns(3)
p1.metric("💰 Total Profit", f"${total_profit:,.0f}")
p2.metric("📊 Average Profit per Order", f"${avg_profit:,.0f}")
p3.metric("📈 Profit Margin", f"{profit_margin:.2f}%")

# Profit by Region
if "Region" in df1.columns:
    st.subheader("🏙️ Profit by Region")
    region_profit = df1.groupby("Region")["Profit"].sum().sort_values(ascending=False)
    st.bar_chart(region_profit)

# -------------------------------
# Customer Lifetime Value (CLV)
st.subheader("👥 Customer Lifetime Value (CLV)")
if "Purchase Address" in df1.columns:
    customer_df = df1.groupby("Purchase Address").agg({
        "sales": "sum",
        "Order ID": "nunique"
    }).rename(columns={"sales": "TotalSales", "Order ID": "TotalOrders"})
    
    aov = customer_df["TotalSales"].sum() / customer_df["TotalOrders"].sum()
    pf = customer_df["TotalOrders"].sum() / customer_df.shape[0]
    clv = aov * pf * (df1["year"].max() - df1["year"].min() + 1)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💳 Avg Order Value", f"${aov:,.2f}")
    c2.metric("🛒 Purchase Freq", f"{pf:.2f}")
    c3.metric("👥 Avg Customer Lifespan", f"{df1['year'].max()-df1['year'].min()+1} yrs")
    c4.metric("💡 CLV", f"${clv:,.2f}")

# -------------------------------
# Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
numeric_cols = df1.select_dtypes(include=["int64","float64"]).columns
if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df1[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# Forecasting with Prophet
st.subheader("🔮 Predictive Sales Forecasting")
forecast_df = df1[["Order Date","sales"]].rename(columns={"Order Date":"ds","sales":"y"}).groupby("ds").sum().reset_index()
m = Prophet(yearly_seasonality=True)
m.fit(forecast_df)
future = m.make_future_dataframe(periods=180)
forecast = m.predict(future)
st.write("📄 Forecast Data Preview", forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(10))
fig1 = m.plot(forecast)
st.pyplot(fig1)
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# -------------------------------
# Export Reports
st.subheader("📥 Download Reports")
excel_buffer = io.BytesIO()
df_filtered.to_excel(excel_buffer, index=False)
st.download_button("⬇️ Download Filtered Data (Excel)",
                   data=excel_buffer,
                   file_name="sales_report.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_buffer = io.BytesIO()
doc = SimpleDocTemplate(pdf_buffer)
styles = getSampleStyleSheet()
story = [
    Paragraph("📊 Sales Report", styles["Title"]),
    Spacer(1, 12),
    Paragraph(f"Total Sales: ${total_sales:,.0f}", styles["Normal"]),
    Paragraph(f"Average Sales: ${avg_sales:,.0f}", styles["Normal"]),
    Paragraph(f"Highest Sale: ${max_sales:,.0f}", styles["Normal"])
]
doc.build(story)
st.download_button("⬇️ Download Sales Summary (PDF)",
                   data=pdf_buffer.getvalue(),
                   file_name="sales_summary.pdf",
                   mime="application/pdf")
