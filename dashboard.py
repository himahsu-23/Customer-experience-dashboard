import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import extract_and_load

st.set_page_config(page_title="📊 Customer Insights Dashboard", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
df1, _ = extract_and_load(
    r"C:\Users\himan\Downloads\archive (1).zip",
    os.path.join("extracted_files", "file1")
)

# -------------------------------
# Data Cleaning & Feature Engineering
# -------------------------------
df1.dropna(inplace=True)

# Convert Order Date
df1["Order Date"] = pd.to_datetime(df1["Order Date"], errors="coerce")
df1 = df1.dropna(subset=["Order Date"])
df1["year"] = df1["Order Date"].dt.year

# Convert numeric columns
df1["Quantity Ordered"] = pd.to_numeric(df1["Quantity Ordered"], errors="coerce")
df1["Price Each"] = pd.to_numeric(df1["Price Each"], errors="coerce")

# Create Sales column
df1["sales"] = df1["Quantity Ordered"] * df1["Price Each"]

# Extract Region from Purchase Address
df1["Region"] = df1["Purchase Address"].apply(
    lambda x: str(x).split(",")[-2].strip() if pd.notnull(x) else None
)

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Filters")

year = st.sidebar.selectbox("Select Year", sorted(df1["year"].unique()))
region = st.sidebar.multiselect("Select Region", df1["Region"].unique())

df_filtered = df1[df1["year"] == year]
if region:
    df_filtered = df_filtered[df_filtered["Region"].isin(region)]

# -------------------------------
# Dashboard Title
# -------------------------------
st.title("📈 Sales Analysis Dashboard")
st.write("This dashboard helps analyze sales trends by year and region.")

# -------------------------------
# Column references
sales_col = "sales"
product_col = "Product"  # CSV me actual naam

# KPI Section
# -------------------------------
st.subheader("📊 Key Metrics")

total_sales = df_filtered["sales"].sum()
avg_sales = df_filtered["sales"].mean()
max_sales = df_filtered["sales"].max()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📊 Average Sales", f"${avg_sales:,.0f}")
col3.metric("🔥 Highest Sale", f"${max_sales:,.0f}")

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("📈 Visualizations")

# Bar Chart - Sales by Region
if not df_filtered.empty:
    st.subheader("📊 Sales by Region (Bar Chart)")
    region_sales = df_filtered.groupby("Region")["sales"].sum().sort_values(ascending=False)
    st.bar_chart(region_sales)

# Line Chart - Yearly Sales Trend
st.subheader("📈 Yearly Sales Trend (Line Chart)")
yearly_sales = df1.groupby("year")["sales"].sum()
st.line_chart(yearly_sales)
# -------------------------------
# Step 5: Monthly Sales Trend
# -------------------------------
st.subheader("📅 Monthly Sales Trend")

if "Order Date" in df1.columns and "sales" in df1.columns:
    # Month column nikal lo
    df1["month"] = df1["Order Date"].dt.to_period("M")

    monthly_sales = df1.groupby("month")["sales"].sum().reset_index()
    monthly_sales["month"] = monthly_sales["month"].astype(str)

    st.line_chart(monthly_sales.set_index("month"))
else:
    st.warning("⚠️ Monthly trend not available (Order Date/Sales column missing).")
# -------------------------------
# -------------------------------
# Step 1: Day-of-Week Sales Pattern
# -------------------------------
st.subheader("📅 Sales Trend by Day of Week")

if "Order Date" in df1.columns and "sales" in df1.columns:
    # Ensure datetime format
    df1["Order Date"] = pd.to_datetime(df1["Order Date"], errors="coerce")

    # Create Day column
    df1["day_of_week"] = df1["Order Date"].dt.day_name()

    # Group by Day of Week
    day_sales = (
        df1.groupby("day_of_week")["sales"]
        .sum()
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    )

    st.bar_chart(day_sales)
else:
    st.warning("⚠️ 'Order Date' or 'sales' column not found for Day-of-Week analysis.")
# -------------------------------
# Step 2: Top 10 Products by Quantity Sold
# -------------------------------
st.subheader("📦 Top 10 Products by Quantity Sold")

if "Product" in df1.columns and "Quantity Ordered" in df1.columns:
    top_products_qty = (
        df1.groupby("Product")["Quantity Ordered"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_products_qty)
else:
    st.warning("⚠️ 'Product' or 'Quantity Ordered' column not found in data.")

# Step 6: Weekly Sales Trend
# -------------------------------
st.subheader("📆 Weekly Sales Trend")

if "Order Date" in df1.columns and "sales" in df1.columns:
    # Week column (ISO week number)
    df1["week"] = df1["Order Date"].dt.strftime("%Y-%U")  # format: 2020-23 (Year-Week)

    weekly_sales = df1.groupby("week")["sales"].sum().reset_index()
    weekly_sales = weekly_sales.sort_values("week")

    st.line_chart(weekly_sales.set_index("week"))
else:
    st.warning("⚠️ Weekly trend not available (Order Date/Sales column missing).")
# -------------------------------
# Step 4: Region-wise Sales (Map + Table)
# -------------------------------
st.subheader("🌍 Region-wise Sales (Map + Table)")

if "Purchase Address" in df1.columns and "sales" in df1.columns:
    # City extract from Purchase Address
    df1["City"] = df1["Purchase Address"].apply(
        lambda x: str(x).split(",")[1].strip() if pd.notnull(x) else None
    )

    city_sales = (
        df1.groupby("City")["sales"]
        .sum()
        .reset_index()
        .sort_values(by="sales", ascending=False)
    )
    # -------------------------------
    # Step 5: Time-Series Analysis (Monthly + Weekly)
    # -------------------------------
    st.subheader("⏳ Time-Series Sales Trend")

    if "Order Date" in df1.columns and "sales" in df1.columns:
        # Ensure datetime
        df1["Order Date"] = pd.to_datetime(df1["Order Date"], errors="coerce")
        df1.dropna(subset=["Order Date"], inplace=True)

        # Monthly Sales
        df1["Month"] = df1["Order Date"].dt.to_period("M")
        monthly_sales = df1.groupby("Month")["sales"].sum()

        st.markdown("### 📅 Monthly Sales Trend")
        st.line_chart(monthly_sales)

        # Weekly Sales
        df1["Week"] = df1["Order Date"].dt.to_period("W")
        weekly_sales = df1.groupby("Week")["sales"].sum()

        st.markdown("### 📆 Weekly Sales Trend")
        st.line_chart(weekly_sales)
    else:
        st.warning("⚠️ 'Order Date' or 'sales' column missing for time-series trend.")
    # -------------------------------
    # Step 6: Day-of-Week Sales Pattern
    # -------------------------------
    st.subheader("📆 Day-of-Week Sales Pattern")

    if "Order Date" in df1.columns and "sales" in df1.columns:
        # Extract day name
        df1["DayOfWeek"] = df1["Order Date"].dt.day_name()
        day_sales = df1.groupby("DayOfWeek")["sales"].sum().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

        st.bar_chart(day_sales)
    else:
        st.warning("⚠️ 'Order Date' or 'sales' column missing for Day-of-Week analysis.")
    # -------------------------------
    # Step 7: Hour-of-Day Sales Trend
    # -------------------------------
    st.subheader("⏰ Hour-of-Day Sales Trend")

    if "Order Date" in df1.columns and "sales" in df1.columns:
        # Extract Hour
        df1["Hour"] = df1["Order Date"].dt.hour
        hour_sales = df1.groupby("Hour")["sales"].sum()

        st.line_chart(hour_sales)
    else:
        st.warning("⚠️ 'Order Date' or 'sales' column missing for Hour-of-Day analysis.")

    # Show top 10 cities in table
    st.write("🏆 Top 10 Cities by Sales")
    st.dataframe(city_sales.head(10))

    # For map we need lat/long → dummy mapping (replace with geocoding if available)
    city_sales["lat"] = np.random.uniform(20, 30, size=len(city_sales))
    city_sales["lon"] = np.random.uniform(70, 90, size=len(city_sales))

    st.map(city_sales[["lat", "lon"]])
else:
    st.warning("⚠️ 'Purchase Address' or 'sales' column not found in data.")


# Pie Chart - Sales Distribution by Region
st.subheader("🥧 Sales Distribution by Region")
if not df_filtered.empty:
    sales_by_region = df_filtered.groupby("Region")["sales"].sum()
    fig, ax = plt.subplots()
    sales_by_region.plot.pie(autopct="%1.1f%%", ax=ax, startangle=90)
    ax.set_ylabel("")
    st.pyplot(fig)
    # -------------------------------
# -------------------------------
# Step 14: Interactive Drill-down (Region -> Product Breakdown)
# -------------------------------
st.subheader("🔎 Drill-down: Region → Products")

if not df_filtered.empty:
    # Step 1: Region select
    selected_region = st.selectbox(
        "Select a Region for Product Breakdown",
        sorted(df_filtered["Region"].dropna().unique())
    )

    region_data = df_filtered[df_filtered["Region"] == selected_region]

    if not region_data.empty:
        # Step 2: Product-wise sales in that region
        product_sales = (
            region_data.groupby("Product")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)  # top 10 products
        )

        st.bar_chart(product_sales)

        # Optional: table bhi show kar do
        st.dataframe(product_sales.reset_index().rename(columns={"sales": "Total Sales"}))
    else:
        st.warning("⚠️ No data available for this region.")
else:
    st.info("Please apply filters above to enable drill-down.")
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
st.subheader("🤖 Product Recommendation (Market Basket Analysis)")
st.info("⚠️ Market Basket Analysis temporarily disabled due to data issues.")

# -------------------------------
# Step 2: Product-wise Analysis
st.subheader("📦 Top 10 Products by Sales")

product_col = "Product"  # <-- actual column name in your CSV

if product_col in df_filtered.columns and "sales" in df_filtered.columns:
    top_products = (
        df_filtered.groupby(product_col)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top_products)
else:
    st.warning("⚠️ Product column not found in data, cannot show Top 10 Products.")
# -------------------------------
# Step 3: Region-wise Sales on Map + Table
# -------------------------------
st.subheader("🌍 Top Regions by Sales")

# Address se City aur State extract karna
if "Purchase Address" in df_filtered.columns and "sales" in df_filtered.columns:
    df_filtered["City"] = df_filtered["Purchase Address"].apply(
        lambda x: str(x).split(",")[1].strip() if pd.notnull(x) and len(str(x).split(",")) > 1 else None
    )
    df_filtered["State"] = df_filtered["Purchase Address"].apply(
        lambda x: str(x).split(",")[-2].strip().split(" ")[0] if pd.notnull(x) and len(str(x).split(",")) > 2 else None
    )

    # City-wise aggregation
    city_sales = (
        df_filtered.groupby("City")["sales"]
        .sum()
        .reset_index()
        .sort_values(by="sales", ascending=False)
    )

    # Top 10 cities
    st.write("🏙️ **Top 10 Cities by Sales**")
    st.dataframe(city_sales.head(10))

    # Chart bhi dikhao
    st.bar_chart(city_sales.set_index("City").head(10))

else:
    st.warning("⚠️ `Purchase Address` ya `sales` column missing hai, map/table generate nahi ho sakta.")

# Step 3: Region-wise Sales on Map
# -------------------------------
# st.subheader("🌍 Region-wise Sales (Map)")
#
# # Agar "Purchase Address" ya "City" column ho to usse location nikalo
# address_col = None
# for c in df_filtered.columns:
#     if "address" in c.lower() or "city" in c.lower() or "state" in c.lower():
#         address_col = c
#         break
#
# if address_col and sales_col in df_filtered.columns:
#     # City/state wise aggregation
#     region_sales = (
#         df_filtered.groupby(address_col)[sales_col]
#         .sum()
#         .reset_index()
#         .sort_values(by=sales_col, ascending=False)
#     )
#
#     st.map(region_sales)  # Streamlit ka built-in map plot
#     st.dataframe(region_sales.head(10))  # Top 10 regions table
# else:
#     st.warning("⚠️ No Address/City/State column found to plot map.")
#


# Preview Data
# -------------------------------
st.subheader("📄 Preview (Filtered Sales Data)")
st.dataframe(df_filtered.head(50))
# -------------------------------
# Step 8: Customer Segmentation (RFM Analysis)
# -------------------------------
st.subheader("👥 Customer Segmentation (RFM Analysis)")

# Assume 'Purchase Address' ~ customer identifier
customer_col = "Purchase Address"

if customer_col in df1.columns and "sales" in df1.columns:
    rfm_df = df1.groupby(customer_col).agg({
        "Order Date": lambda x: (df1["Order Date"].max() - x.max()).days,  # Recency
        "Order ID": "count",   # Frequency
        "sales": "sum"         # Monetary
    }).reset_index()

    rfm_df.columns = ["Customer", "Recency", "Frequency", "Monetary"]

    # Simple segmentation rule
    rfm_df["Segment"] = pd.cut(
        rfm_df["Monetary"],
        bins=[-1, 1000, 5000, 20000, 100000],
        labels=["Low Spender", "Mid Spender", "High Spender", "VIP"]
    )

    st.dataframe(rfm_df.head(20))

    # Bar chart - Segment distribution
    seg_counts = rfm_df["Segment"].value_counts().sort_index()
    st.bar_chart(seg_counts)

else:
    st.warning("⚠️ Customer column not found for RFM (need Purchase Address + Sales).")
# -------------------------------
# Step 9: Correlation Heatmap
# -------------------------------
import seaborn as sns

st.subheader("📊 Correlation Heatmap (Numeric Features)")

numeric_cols = df1.select_dtypes(include=["int64", "float64"]).columns

if len(numeric_cols) > 1:
    corr = df1[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ Not enough numeric columns to show correlation heatmap.")
# -------------------------------
# Step 9b: Scatter Plots (Sales vs Other Features)
# -------------------------------
st.subheader("📉 Scatter Plots (Sales vs Features)")

if "sales" in df1.columns:
    col_x = st.selectbox("Select feature for X-axis", ["Quantity Ordered", "Price Each"])

    if col_x in df1.columns:
        fig, ax = plt.subplots()
        ax.scatter(df1[col_x], df1["sales"], alpha=0.5)
        ax.set_xlabel(col_x)
        ax.set_ylabel("Sales")
        ax.set_title(f"Sales vs {col_x}")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ Column {col_x} not found in data.")
else:
    st.warning("⚠️ 'sales' column not found in data.")
# -------------------------------
# Step 10: Profit Calculation & Visualization
# -------------------------------
st.subheader("💹 Profit Analysis")

# Step 1: Identify Cost Price column (agar h)
cost_candidates = [c for c in df1.columns if "cost" in c.lower() or "base" in c.lower()]
if cost_candidates:
    cost_col = cost_candidates[0]
    df1["Cost Price"] = pd.to_numeric(df1[cost_col], errors="coerce")
else:
    # Assumption: Cost = 70% of Price Each
    df1["Cost Price"] = df1["Price Each"] * 0.7

# Step 2: Profit column
df1["Profit"] = (df1["Price Each"] - df1["Cost Price"]) * df1["Quantity Ordered"]

# Step 3: KPI for profit
total_profit = df1["Profit"].sum()
avg_profit = df1["Profit"].mean()
profit_margin = (df1["Profit"].sum() / df1["sales"].sum()) * 100

p1, p2, p3 = st.columns(3)
p1.metric("💰 Total Profit", f"${total_profit:,.0f}")
p2.metric("📊 Average Profit per Order", f"${avg_profit:,.0f}")
p3.metric("📈 Profit Margin", f"{profit_margin:.2f}%")

# Step 4: Visualization - Profit by Region
if "Region" in df1.columns:
    st.subheader("🏙️ Profit by Region")
    region_profit = df1.groupby("Region")["Profit"].sum().sort_values(ascending=False)
    st.bar_chart(region_profit)

# Step 5: Visualization - Profit Trend by Year
if "year" in df1.columns:
    st.subheader("📆 Profit Trend by Year")
    yearly_profit = df1.groupby("year")["Profit"].sum()
    st.line_chart(yearly_profit)
# -------------------------------
# Step 11: Customer Lifetime Value (CLV) Estimation
# -------------------------------
st.subheader("👥 Customer Lifetime Value (CLV)")

# Step 1: Identify Customer column (usually Purchase Address or Customer ID)
cust_candidates = [c for c in df1.columns if "customer" in c.lower() or "address" in c.lower() or "id" in c.lower()]
if cust_candidates:
    customer_col = cust_candidates[0]
else:
    customer_col = "Purchase Address"  # fallback

# Step 2: Group by customer
customer_df = df1.groupby(customer_col).agg({
    "sales": "sum",
    "Order ID": "nunique"
}).rename(columns={"sales": "TotalSales", "Order ID": "TotalOrders"})

# Step 3: Metrics
aov = customer_df["TotalSales"].sum() / customer_df["TotalOrders"].sum()  # Avg Order Value
pf = customer_df["TotalOrders"].sum() / customer_df.shape[0]              # Purchase Frequency
cv = aov * pf                                                             # Customer Value

# Estimate Customer Lifespan (based on year span in dataset)
if "year" in df1.columns:
    acl = df1["year"].max() - df1["year"].min() + 1
else:
    acl = 2  # assumption: 2 years

clv = cv * acl

# Step 4: Show KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("💳 Avg Order Value (AOV)", f"${aov:,.2f}")
c2.metric("🛒 Purchase Freq (PF)", f"{pf:.2f}")
c3.metric("👥 Avg Customer Lifespan", f"{acl} years")
c4.metric("💡 CLV (Estimated)", f"${clv:,.2f}")

# Step 5: Visualization - Top 10 Customers by CLV
st.subheader("🏆 Top 10 Customers by Sales (Proxy for CLV)")
top_customers = customer_df.sort_values("TotalSales", ascending=False).head(10)
st.bar_chart(top_customers["TotalSales"])
# -------------------------------
# Step 12: Predictive Sales Forecasting (Prophet)
# -------------------------------
st.subheader("🔮 Predictive Sales Forecasting (Prophet)")

from prophet import Prophet

# Step 1: Prepare data
forecast_df = df1.copy()
forecast_df = forecast_df[["Order Date", "sales"]].rename(columns={"Order Date": "ds", "sales": "y"})
forecast_df = forecast_df.groupby("ds").sum().reset_index()

# Step 2: Train Prophet model
m = Prophet(yearly_seasonality=True, daily_seasonality=False)
m.fit(forecast_df)

# Step 3: Future dataframe (next 6 months prediction)
future = m.make_future_dataframe(periods=180)  # 180 days ≈ 6 months
forecast = m.predict(future)

# Step 4: Plot
fig1 = m.plot(forecast)
st.pyplot(fig1)

# Step 5: Components (trend + seasonality)
st.subheader("📊 Forecast Components (Trend/Seasonality)")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# Show data
st.write("📄 Forecast Data Preview", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
# -------------------------------
# Step 13: Export as Excel/PDF
# -------------------------------
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.subheader("📥 Download Reports")

# Excel export
excel_buffer = io.BytesIO()
df_filtered.to_excel(excel_buffer, index=False)
st.download_button(
    label="⬇️ Download Filtered Data (Excel)",
    data=excel_buffer,
    file_name="sales_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# PDF export
pdf_buffer = io.BytesIO()
doc = SimpleDocTemplate(pdf_buffer)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("📊 Sales Report", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph(f"Total Sales: {total_sales:,.0f}", styles["Normal"]))
story.append(Paragraph(f"Average Sales: {avg_sales:,.0f}", styles["Normal"]))
story.append(Paragraph(f"Highest Sale: {max_sales:,.0f}", styles["Normal"]))

doc.build(story)
st.download_button(
    label="⬇️ Download Sales Summary (PDF)",
    data=pdf_buffer.getvalue(),
    file_name="sales_summary.pdf",
    mime="application/pdf"
)
