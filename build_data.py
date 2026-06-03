import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / 'data' / '1850-2025.csv')

# Convert the 'Date' field from int YYYYMM to datetime
date_time_df = df.copy()
date_time_df['Date'] = pd.to_datetime(date_time_df['Date'].astype(str), format='%Y%m')


AnomalyData = date_time_df.copy()
AnomalyData.set_index('Date', inplace=True)

#group data for a century
century = AnomalyData.loc["1925-01-01":"2025-04-01"]

# Calculate the rolling mean with a window of 12 months (1 year)
rolling_mean_year = century['Anomaly'].rolling(window=12, center=True).mean()
# Calculate the rolling mean with a window of 60 months (5 years)
rolling_mean_5years = century['Anomaly'].rolling(window=60, center=True).mean()
# Calculate the rolling mean with a window of 120 months (10 years)
rolling_mean_10years = century['Anomaly'].rolling(window=120, center=True).mean()

#get summary statistics for the century data
century_summary_stats = century.describe()

century.to_csv('data/century_anomalies.csv', index=True)
rolling_mean_year.to_csv('data/rolling_mean_year.csv', index=True)
rolling_mean_5years.to_csv('data/rolling_mean_5years.csv', index=True)
rolling_mean_10years.to_csv('data/rolling_mean_10years.csv', index=True)
