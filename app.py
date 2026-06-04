"""Global Temperature Anomalies — an editorial data-story dashboard.

Read-only: every number on this page comes from the CSVs that `build_data.py`
writes. Run `python build_data.py` first; this app never computes or writes data.
"""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

DATA_DIR = Path(__file__).parent / "data"
FORECAST_END = 2045

# --- Warming-stripes palette (Ed Hawkins, cool blue -> warm red) ----------------
STRIPES = ["#08306b", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
           "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]

# --- Per-model line colours -----------------------------------------------------
C_OBSERVED  = "#b9b5ac"   # faint warm grey
C_LINEAR    = "#5a6470"   # slate, the baseline
C_PIECEWISE = "#b2182b"   # accent red, the CV winner
C_QUAD      = "#c99a3f"   # muted amber, flagged risky
C_ROLL      = ["#92c5de", "#4393c3", "#2166ac"]  # 1 / 5 / 10 year, graded blues

INK, INK_FAINT, HAIR = "#17191c", "#8c8d86", "#e4ded4"
FONT_SANS = "Geist, system-ui, sans-serif"


# ============================================================================
# Data loading (read-only)
# ============================================================================
def load() -> dict:
    rd = lambda name: pd.read_csv(DATA_DIR / name)
    data = {
        "century":  rd("century_anomalies.csv"),
        "roll_1":   rd("rolling_mean_year.csv"),
        "roll_5":   rd("rolling_mean_5years.csv"),
        "roll_10":  rd("rolling_mean_10years.csv"),
        "fits":     rd("yearly_fits.csv"),
        "forecast": rd("forecast.csv"),
        "cv":       rd("cv_scores.csv"),
        "summary":  rd("summary.csv").iloc[0],
    }
    for key in ("century", "roll_1", "roll_5", "roll_10"):
        data[key]["Date"] = pd.to_datetime(data[key]["Date"])
    return data


# ============================================================================
# Plotly theming + chart helpers
# ============================================================================
def base_layout(yaxis_title: str, height: int = 360) -> go.Layout:
    """Shared chart theme so every figure reads from one place."""
    axis = dict(showgrid=True, gridcolor=HAIR, gridwidth=1, zeroline=False,
                linecolor=HAIR, ticks="outside", tickcolor=HAIR,
                tickfont=dict(color=INK_FAINT, size=12))
    return go.Layout(
        height=height,
        margin=dict(l=54, r=18, t=14, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, size=13, color=INK),
        xaxis={**axis, "title": None},
        yaxis={**axis, "title": dict(text=yaxis_title, font=dict(color=INK_FAINT, size=12))},
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )


def graph(fig: go.Figure, gid: str) -> dcc.Graph:
    return dcc.Graph(id=gid, figure=fig, config={"displayModeBar": False},
                     style={"width": "100%"})


def line(x, y, name, color, width=2, dash=None, mode="lines"):
    return go.Scatter(x=x, y=y, name=name, mode=mode,
                      line=dict(color=color, width=width, dash=dash),
                      marker=dict(color=color, size=5))


# ---- Hero warming-stripes bar (real data, one span per year) -------------------
def anomaly_to_color(value, lo, hi) -> str:
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    pos = frac * (len(STRIPES) - 1)
    i = int(pos)
    if i >= len(STRIPES) - 1:
        return STRIPES[-1]
    t = pos - i
    a = STRIPES[i].lstrip("#"); b = STRIPES[i + 1].lstrip("#")
    rgb = [round(int(a[k:k+2], 16) * (1 - t) + int(b[k:k+2], 16) * t) for k in (0, 2, 4)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def stripes_bar(fits: pd.DataFrame) -> html.Div:
    lo, hi = fits["Anomaly"].min(), fits["Anomaly"].max()
    spans = [html.Span(style={"background": anomaly_to_color(v, lo, hi)})
             for v in fits["Anomaly"]]
    years = fits["Year"]
    return html.Div([
        html.Div(spans, className="stripes"),
        html.Div([html.Span(str(int(years.iloc[0]))),
                  html.Span("each stripe is one year, coloured by its anomaly"),
                  html.Span(str(int(years.iloc[-1])))], className="stripes-caption"),
    ])


# ============================================================================
# Sections
# ============================================================================
def hero(d) -> html.Section:
    s = d["summary"]
    return html.Section(className="hero", children=html.Div(className="wrap", children=[
        html.P("Global surface temperature · 1925-2025", className="kicker"),
        html.H1("How fast is the Earth warming, and which model should we trust to forecast it?"),
        html.Div(className="stat-row", children=[
            html.Span(f"+{s['warming_per_decade']:.2f}°C", className="stat-big"),
            html.Span("per decade", className="stat-unit"),
            html.Span(f"95% CI  {s['ci_low_per_decade']:.2f} to {s['ci_high_per_decade']:.2f}",
                      className="stat-ci"),
        ]),
        html.P("A century of monthly readings, three trend models, and a test of how well each "
               "forecasts years it was never trained on. The model that fits the past most "
               "closely is not the one that predicts it best.", className="lead"),
        stripes_bar(d["fits"]),
    ]))


def signal_section(d) -> html.Section:
    raw = go.Figure(layout=base_layout("Anomaly (°C)", height=300))
    raw.add_trace(line(d["century"]["Date"], d["century"]["Anomaly"],
                       "Monthly anomaly", C_OBSERVED, width=1))

    roll = go.Figure(layout=base_layout("Anomaly (°C)", height=300))
    for df, name, col in [(d["roll_1"], "1-year mean", C_ROLL[0]),
                          (d["roll_5"], "5-year mean", C_ROLL[1]),
                          (d["roll_10"], "10-year mean", C_ROLL[2])]:
        roll.add_trace(line(df["Date"], df["Anomaly"], name, col,
                            width=2 if name != "10-year mean" else 3))

    return html.Section(className="reveal", children=html.Div(className="wrap", children=[
        html.P("The signal", className="kicker"),
        html.H2("Reading the trend out of the noise"),
        html.P("Month to month the series is noisy. Widening the averaging window from one year "
               "to ten strips out much of that short-term variation and leaves the longer-term "
               "signal: a rise sustained across decades that no single warm year can explain.",
               className="lead"),
        html.Div(className="chart-grid", children=[
            html.Div(className="panel full", children=[graph(raw, "raw")]),
            html.Div(className="panel full", children=[graph(roll, "rolling")]),
        ]),
        html.P("Raw monthly anomalies (top); the same series under 1, 5 and 10-year rolling "
               "means (bottom). Source: NOAA NCEI, measured against the 1901 to 2000 average.",
               className="caption"),
    ]))


def fitting_section(d) -> html.Section:
    fits = d["fits"]
    fig = go.Figure(layout=base_layout("Anomaly (°C)", height=420))
    fig.add_trace(line(fits["Year"], fits["Anomaly"], "Observed", C_OBSERVED,
                       mode="markers"))
    fig.add_trace(line(fits["Year"], fits["Predicted_linear"], "Linear", C_LINEAR))
    fig.add_trace(line(fits["Year"], fits["Predicted_quad"], "Quadratic", C_QUAD))
    fig.add_trace(line(fits["Year"], fits["Predicted_piecewise"], "Piecewise (1979)", C_PIECEWISE, width=3))

    return html.Section(className="reveal", children=html.Div(className="wrap", children=[
        html.P("Fitting the trend", className="kicker"),
        html.H2("Fitting the past is not the same as forecasting it"),
        html.P("A straight line, a quadratic curve, and a piecewise line that bends in 1979 each "
               "describe the century. On the data they were trained on, the more flexible models "
               "fit more closely, but that closeness is easy to mistake for accuracy.",
               className="lead"),
        html.Div(className="panel full", children=[graph(fig, "fits")]),
        html.P("Models fitted to the full 1925 to 2025 record. A closer fit here is not evidence "
               "of a better forecast, which is what the held-out test below sets out to measure.",
               className="caption"),
        cv_table(d["cv"]),
        html.P("Time-series cross-validation trains each model on earlier years and scores it on "
               "later years it has not seen, then widens the window and repeats. The ranking "
               "reverses. The piecewise model, narrowly behind on the full fit, forecasts unseen "
               "years best; the quadratic, the closest in-sample fit, does worst out of sample.",
               className="caption"),
    ]))


def cv_table(cv: pd.DataFrame) -> html.Table:
    best_idx = cv["CV_MAE"].idxmin()
    verdicts = {
        "Linear":    "An honest baseline at a steady rate.",
        "Quadratic": "Closest fit to the past, weakest out of sample.",
        "Piecewise": "Forecasts unseen years best.",
    }
    rows = []
    for i, r in cv.iterrows():
        is_best = i == best_idx
        name_cell = [r["Model"]]
        if is_best:
            name_cell.append(html.Span("best", className="tag"))
        rows.append(html.Tr(className="best" if is_best else "", children=[
            html.Td(name_cell),
            html.Td(f"{r['CV_MAE']:.3f}"),
            html.Td(f"{r['CV_RMSE']:.3f}"),
            html.Td(verdicts.get(r["Model"], ""), className="verdict"),
        ]))
    return html.Table(className="cv", children=[
        html.Thead(html.Tr([html.Th("Model"), html.Th("CV MAE"),
                            html.Th("CV RMSE"), html.Th("Verdict", className="verdict")])),
        html.Tbody(rows),
    ])


def forecast_section(d) -> html.Section:
    fits, fc = d["fits"], d["forecast"]
    last = fits.iloc[-1]
    yr0 = int(last["Year"])

    # anchor each forecast line at the model's final fitted value, then extend to 2045
    def fwd(col_fit, col_fc):
        return [yr0] + fc["Year"].tolist(), [last[col_fit]] + fc[col_fc].tolist()

    fig = go.Figure(layout=base_layout("Anomaly (°C)", height=440))
    fig.add_trace(line(fits["Year"], fits["Anomaly"], "Observed", C_OBSERVED, mode="markers"))
    lx, ly = fwd("Predicted_linear", "Linear_future")
    px, py = fwd("Predicted_piecewise", "Piecewise_future")
    qx, qy = fwd("Predicted_quad", "Quad_future")
    fig.add_trace(line(lx, ly, "Linear", C_LINEAR, width=2.5))
    fig.add_trace(line(px, py, "Piecewise (1979)", C_PIECEWISE, width=3))
    fig.add_trace(line(qx, qy, "Quadratic (risky)", C_QUAD, width=2.5, dash="dot"))

    fig.add_annotation(x=fc["Year"].iloc[-1], y=fc["Quad_future"].iloc[-1],
                       text="unreliable extrapolation", showarrow=True, arrowhead=0,
                       arrowcolor=C_QUAD, ax=-60, ay=-28,
                       font=dict(color=C_QUAD, size=11, family=FONT_SANS))

    def vis(mask):  # Observed always on
        return [True] + mask
    fig.update_layout(updatemenus=[dict(
        type="dropdown", direction="down", x=1, xanchor="right", y=1.18, yanchor="top",
        bgcolor="#faf8f5", bordercolor=HAIR, font=dict(family=FONT_SANS, size=12),
        buttons=[
            dict(label="All models", method="update", args=[{"visible": vis([True, True, True])}]),
            dict(label="Linear + Piecewise", method="update", args=[{"visible": vis([True, True, False])}]),
            dict(label="Piecewise only", method="update", args=[{"visible": vis([False, True, False])}]),
            dict(label="Linear only", method="update", args=[{"visible": vis([True, False, False])}]),
            dict(label="Quadratic (risky)", method="update", args=[{"visible": vis([False, False, True])}]),
        ])])

    s = d["summary"]
    return html.Section(className="reveal", children=html.Div(className="wrap", children=[
        html.P("Looking ahead", className="kicker"),
        html.H2(f"Projecting to {FORECAST_END}, with the caveats kept in view"),
        html.P(f"Each model is extended to {FORECAST_END} and shown side by side on purpose. The "
               "quadratic is drawn dotted and labelled, because the curvature that fit the past "
               "so closely also sends it climbing fastest into years it has no support for.",
               className="lead"),
        html.Div(className="panel full", children=[graph(fig, "forecast")]),
        html.P(f"Forecasts 2026 to {FORECAST_END}. How far to trust each line follows the "
               f"held-out scores above: the piecewise (CV MAE {s['best_cv_mae']:.3f}) is the most "
               "defensible here and the quadratic the least. Use the menu to isolate a model.",
               className="caption"),
    ]))


def footer() -> html.Footer:
    return html.Footer(className="colophon", children=html.Div(className="wrap", children=[
        html.P("Methods & sources", className="kicker"),
        html.Ul([
            html.Li("Data: NOAA National Centers for Environmental Information (NCEI), "
                    "monthly global anomalies measured against the 1901 to 2000 average."),
            html.Li("Models: ordinary least squares (linear, quadratic, piecewise with a 1979 knot), "
                    "scored with expanding-window time-series cross-validation."),
            html.Li("Pipeline: every figure here is generated by build_data.py; this app only reads."),
        ]),
        html.P("Built by Muhammed Panjwani", className="by"),
    ]))


# ============================================================================
# App
# ============================================================================
def build_layout(d) -> html.Div:
    return html.Div([
        hero(d), signal_section(d), fitting_section(d), forecast_section(d), footer(),
    ])


FONTS = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&"
         "family=Geist+Mono:wght@400;500&"
         "family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap")

app = Dash(__name__, external_stylesheets=[FONTS],
           title="Global Temperature Anomalies")
app.index_string = app.index_string.replace(
    "<head>",
    '<head>\n    <link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')

app.layout = build_layout(load())
server = app.server

if __name__ == "__main__":
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=debug, port=port)
