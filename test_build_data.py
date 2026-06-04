"""Plumbing tests for the data pipeline. Run: `python test_build_data.py` (or pytest)."""

from pathlib import Path

import build_data as bd

DATA_DIR = Path(__file__).parent / "data"
MODELS = {"Linear", "Quadratic", "Piecewise"}


def test_yearly_fits_complete():
    f = bd.yearly_fits
    assert list(f["Year"]) == list(range(1925, 2026)), "expected one row per year 1925-2025"
    for col in ["Anomaly", "Predicted_linear", "Predicted_quad", "Predicted_piecewise"]:
        assert col in f.columns, f"missing column {col}"
        assert f[col].notna().all(), f"NaN found in {col}"


def test_forecast_years_and_values():
    fc = bd.forecast
    assert list(fc["Year"]) == list(range(2026, 2046)), "forecast must cover exactly 2026-2045"
    for col in ["Linear_future", "Quad_future", "Piecewise_future"]:
        assert fc[col].notna().all(), f"NaN in forecast column {col}"


def test_cv_scores_all_models_positive():
    cv = bd.cv_scores
    assert set(cv["Model"]) == MODELS, "CV table must score all three models"
    assert (cv["CV_MAE"] > 0).all() and (cv["CV_RMSE"] > 0).all()


def test_summary_headline():
    s = bd.summary.iloc[0]
    assert s["warming_per_decade"] > 0, "a warming planet should report a positive rate"
    assert s["ci_low_per_decade"] <= s["warming_per_decade"] <= s["ci_high_per_decade"]
    assert s["best_model"] in MODELS


def test_pipeline_is_self_contained(tmp_path=None):
    """save_outputs writes every file the app needs, from the raw input alone."""
    out = tmp_path or (DATA_DIR / "_test_out")
    out.mkdir(exist_ok=True)
    bd.save_outputs(bd.century, bd.rolling_means, yearly_fits=bd.yearly_fits,
                    forecast=bd.forecast, cv_scores=bd.cv_scores, summary=bd.summary,
                    out_dir=out)
    needed = ["century_anomalies.csv", "rolling_mean_year.csv", "rolling_mean_5years.csv",
              "rolling_mean_10years.csv", "yearly_fits.csv", "forecast.csv",
              "cv_scores.csv", "summary.csv"]
    missing = [n for n in needed if not (out / n).exists()]
    assert not missing, f"save_outputs did not write: {missing}"
    if out.name == "_test_out":  # clean up our own scratch dir
        for n in needed:
            (out / n).unlink(missing_ok=True)
        out.rmdir()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} passed")
