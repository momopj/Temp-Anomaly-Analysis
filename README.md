# Global Temperature Anomaly Analysis

## Overview

**How have global temperature anomalies changed over the past 100 years?**  
This project explores long-term trends in global temperature anomalies using historical data and forecasting techniques.

A **temperature anomaly** is the difference between an observed temperature and a baseline average, making it ideal for detecting climate trends over time.

---

## Data Source

The dataset used comes from the **[National Centers for Environmental Information (NCEI)](https://www.ncei.noaa.gov/)**, which provides reliable climate-related data.

---

## Project Structure

### `Analysis.ipynb`

A complete Jupyter Notebook walking through the entire data science pipeline:
- Data loading  
- Data cleaning  
- Exploratory data analysis (EDA)  
- Statistical modeling  
- Future predictions using:
  - Linear Regression  
  - Polynomial Regression  
  - Prophet (a time-series forecasting model by Meta)  
- Clear explanations and comments throughout

---

### `app.py`

An interactive **Plotly Dash dashboard** to visualize:
- Raw and smoothed temperature anomaly trends
- Regression lines and future forecasts
- Model comparisons
- A forecast summary table (2040–2045)

---

## Dashboard Features

- Side-by-side visualizations for rolling averages
- Forecast dropdown to toggle between models
- Table of predictions from 2040 to 2045
- Clean, styled layout using Plotly and Dash

---

## Getting Started

To run the dashboard locally:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/momopj/Temp-Anomaly-Analysis.git
   cd Temp-Anomaly_Analysis
2. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
4. **Run the dashboard**
   ```bash
   python app.py
5. **Open the dashboard in your browser**
Copy the URL in your terminal and paste it into your browser

---

## Dashboard Preview

![Dashboard](preview.png)

## Future work
I wish to do more work on this project by not only working with mean yearly data to predict future anomalies to see how the timespan affects the predictions.

I would also like to use other models such as ARIMA for time series forecasting.

Overall its been fun working on this project and I'd like to do many more in the future, Ive learnt a lot from doing this and found the data very interesting and beginner friendly!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.