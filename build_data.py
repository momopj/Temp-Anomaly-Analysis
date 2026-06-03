
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

DATA_DIR = Path(__file__).parent / "data"
PERIOD = ("1925-01-01", "2025-04-01")


def load_data(path=DATA_DIR / "1850-2025.csv"):
    """Load the anomaly CSV and return it indexed by datetime."""
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"].astype(str), format="%Y%m")
    return data.set_index("Date")


def slice_period(data, start=PERIOD[0], end=PERIOD[1]):
    """Return the slice of data covering the period of interest."""
    return data.loc[start:end]


def compute_rolling_means(century):
    """Centred rolling means of the monthly anomaly at 1, 5 and 10 years."""
    anomaly = century["Anomaly"]
    return {
        "year": anomaly.rolling(window=12, center=True).mean(),
        "5_year": anomaly.rolling(window=60, center=True).mean(),
        "10_year": anomaly.rolling(window=120, center=True).mean(),
    }


def prepare_yearly(century):
    """Resample to annual means and add the centred year features."""
    yearly = century["Anomaly"].resample("YE").mean().reset_index()
    yearly["Year"] = yearly["Date"].dt.year
    yearly["Year_centred"] = yearly["Year"] - yearly["Year"].mean()
    yearly["Year_centred_sq"] = yearly["Year_centred"] ** 2
    return yearly


def fit_linear(yearly):
    """Fit the linear model. Adds 'Predicted_linear'; returns (sklearn, statsmodels)."""
    X = yearly["Year_centred"].to_numpy().reshape(-1, 1)
    y = yearly["Anomaly"].to_numpy()

    sk_model = LinearRegression().fit(X, y)
    yearly["Predicted_linear"] = sk_model.predict(X)

    sm_model = sm.OLS(yearly["Anomaly"], sm.add_constant(yearly["Year_centred"])).fit()
    return sk_model, sm_model


def fit_quadratic(yearly):
    """Fit the quadratic model. Adds 'Predicted_poly'; returns (sklearn, statsmodels)."""
    X = yearly[["Year_centred", "Year_centred_sq"]].to_numpy()
    y = yearly["Anomaly"].to_numpy()

    sk_model = LinearRegression().fit(X, y)
    yearly["Predicted_poly"] = sk_model.predict(X)

    sm_model = sm.OLS(
        yearly["Anomaly"],
        sm.add_constant(yearly[["Year_centred", "Year_centred_sq"]]),
    ).fit()
    return sk_model, sm_model


def compare_models(linear_ols, quad_ols):
    """Comparison table of fit and diagnostic metrics for the two models."""
    return pd.DataFrame(
        {
            "Linear": [linear_ols.rsquared, linear_ols.rsquared_adj,
                       linear_ols.aic, linear_ols.bic, durbin_watson(linear_ols.resid)],
            "Quadratic": [quad_ols.rsquared, quad_ols.rsquared_adj,
                          quad_ols.aic, quad_ols.bic, durbin_watson(quad_ols.resid)],
        },
        index=["R-squared", "Adj. R-squared", "AIC", "BIC", "Durbin-Watson"],
    )


def predict_linear(sk_model, year, mean_year):
    """Predict the anomaly for a calendar year using the centred linear model."""
    return sk_model.predict([[year - mean_year]])[0]

def predict_quadratic(sk_model, year, mean_year):
    """Predict the anomaly for a calendar year using the centred quadratic model."""
    c = year - mean_year
    return sk_model.predict([[c, c ** 2]])[0]


def warming_rate(quad_ols, year, mean_year):
    """Warming rate (°C per decade) at a year, from the quadratic."""
    b1 = quad_ols.params["Year_centred"]
    b2 = quad_ols.params["Year_centred_sq"]
    c = year - mean_year
    return (b1 + 2 * b2 * c) * 10


def save_outputs(century, rolling, out_dir=DATA_DIR):
    """Write the century slice and rolling means to CSV."""
    century.to_csv(out_dir / "century_anomalies.csv")
    rolling["year"].to_csv(out_dir / "rolling_mean_year.csv")
    rolling["5_year"].to_csv(out_dir / "rolling_mean_5years.csv")
    rolling["10_year"].to_csv(out_dir / "rolling_mean_10years.csv")


# ---- assemble the standard objects so they import by name ----
df = load_data()
century = slice_period(df)
century_summary_stats = century.describe()

rolling_means = compute_rolling_means(century)
# backwards-compatible aliases for your existing notebook references
rolling_mean_year = rolling_means["year"]
rolling_mean_5years = rolling_means["5_year"]
rolling_mean_10years = rolling_means["10_year"]

yearly_df = prepare_yearly(century)
mean_year = yearly_df["Year"].mean()

model1, ols = fit_linear(yearly_df)
model1_slope = model1.coef_[0]
model1_intercept = model1.intercept_
ci_low, ci_high = ols.conf_int().loc["Year_centred"]
durbin_watson_stat = durbin_watson(ols.resid)

linear_pred_2040 = predict_linear(model1, 2040, mean_year)
linear_pred_2024 = predict_linear(model1, 2024, mean_year)
actual_2024 = yearly_df.loc[yearly_df["Year"] == 2024, "Anomaly"].values[0]

model2, ols2 = fit_quadratic(yearly_df)
comparison = compare_models(ols, ols2)

quad_pred_2040 = predict_quadratic(model2, 2040, mean_year)
quad_pred_2024 = predict_quadratic(model2, 2024, mean_year)

rate_1925 = warming_rate(ols2, 1925, mean_year)
rate_1975 = warming_rate(ols2, 1975, mean_year)
rate_2025 = warming_rate(ols2, 2025, mean_year)


if __name__ == "__main__":
    save_outputs(century, rolling_means)
    print(yearly_df.head(), "\n")
    print("Model comparison:")
    print(comparison.round(4))
    print(f"\nLinear 2040: {linear_pred_2040:.3f} °C  |  "
          f"2024 pred: {linear_pred_2024:.3f} °C  |  actual: {actual_2024:.3f} °C")

