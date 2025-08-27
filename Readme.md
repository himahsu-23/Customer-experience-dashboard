📈 Sales Analysis Dashboard
This project is a comprehensive sales analysis dashboard built with Python and Streamlit. It helps in visualizing key sales metrics, identifying trends, and performing in-depth customer and product analysis.
✨ Features
Interactive Dashboard: A web-based dashboard with filters for year and region to explore data.
Key Performance Indicators (KPIs): Displays crucial metrics such as total sales, average sales, and the highest single sale.
Data Visualization: Includes various charts to visualize sales trends by region, day of the week, hour of the day, and product.
Time-Series Analysis: Visualizes monthly and weekly sales trends to identify seasonality and long-term patterns.
Predictive Forecasting: Uses the Prophet library to forecast future sales trends and visualize forecast components.
Customer Segmentation: Performs a simple RFM (Recency, Frequency, Monetary) analysis to segment customers.
Data Export: Allows users to download filtered data in Excel format and a sales summary report in PDF format.
Utility Functions: Contains a helper script to automatically extract and load data from compressed .zip files.
📂 Project Structure
dashboard.py: The main script for the Streamlit web application. It handles data processing, visualization, and user interface components.
utils.py: A helper module containing the extract_and_load function, which is used for handling zip file extraction and initial data loading/cleaning.
main.py: A simple script used for testing the utility functions and verifying data loading.
config.py: A configuration file to manage a list of data file paths.
🛠️ Installation
Clone the repository:
git clone https://your-repository-url.git
cd sales-analysis-dashboard


Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required libraries:
pip install streamlit pandas matplotlib seaborn prophet reportlab

Note: The prophet library might require pystan to be installed first. If you face issues, refer to the pystan documentation.
🚀 Usage
Place your data files: Ensure your .zip data files are placed in the paths specified in config.py.
Run the Streamlit application:
streamlit run dashboard.py


The application will open in your default web browser. You can now use the filters on the left sidebar and interact with the various visualizations.
💡 Future Enhancements
Dynamic Data Loading: Modify the application to allow users to upload their own data files directly.
Refined Data Export: Add more customization options for the PDF and Excel reports.
Market Basket Analysis: Re-enable and refine the Market Basket Analysis feature for product recommendations.
Geographical Mapping: Enhance the Region-wise Sales section with more accurate geographical data to visualize sales on a map.
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
