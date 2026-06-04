# Global Temperature Anomaly Analysis

**How fast has the Earth warmed over the past century, and how well can simple models forecast that warming?**

This project works through a century of global temperature anomalies, from exploring the raw
record to fitting and comparing trend models, and presents the result as an interactive
dashboard. A temperature anomaly is the difference between an observed temperature and a
long-term baseline, which makes it well suited to picking out climate trends rather than
absolute temperatures.

The headline figure is a warming rate of about **0.11°C per decade** over 1925 to 2025, but the
more interesting result is what happens when the models are asked to forecast rather than
describe: the model that fits the past most closely is not the one that predicts best.

preview/Screenshot 2026-06-04 at 14.07.58.png

---

## Data

The data comes from the **[National Centers for Environmental Information (NCEI)](https://www.ncei.noaa.gov/)**,
as monthly global anomalies measured against the 1901 to 2000 average. The raw file
(`data/1850-2025.csv`) is the only input that is not generated; everything else in `data/` is
derived from it.

---

## How it works

The project is split into two jobs that used to be tangled together.

- **`build_data.py` is the pipeline.** It reads the raw NCEI file and produces every derived
  file the rest of the project uses: the century slice, the rolling means, the fitted models,
  the forecasts, and the cross-validation scores. It is the single source of truth for the
  numbers. Run it once and every figure downstream is rebuilt from scratch:

  ```bash
  python build_data.py
  ```

  If you delete everything in `data/` except the raw input and run that command, the dashboard
  still works. That is the test that the pipeline is genuinely reproducible.

- **`Analysis.ipynb` is the write-up.** It imports the pipeline and narrates the exploratory
  analysis and the modelling, explaining each step and why it was taken.

- **`app.py` only reads.** The dashboard loads the derived files and displays them. It computes
  nothing and writes nothing.

---

## The models, and what they show

Three trend models are fitted to the yearly anomalies, each one addressing a weakness in the
last:

- **Linear** gives a single average rate of warming, about 0.11°C per decade. It is the honest
  baseline, but its residuals show a pattern it cannot explain: the warming has not been steady.
- **Quadratic** adds a squared term so the line can curve, which lets it follow the accelerating
  rise. It fits the past far more closely, with the rate climbing from near zero in the 1920s to
  around 0.28°C per decade today.
- **Piecewise** reads the same shape a different way, as a gentle rise before a chosen breakpoint
  in 1979 and a steeper one after it.

On the data they were trained on, the two flexible models clearly beat the straight line. But a
close fit to the past is not proof of a good forecast. **Time-series cross-validation** tests
that directly, training each model on earlier years and scoring it on later years it has not
seen. The ranking reverses: the piecewise model forecasts unseen years best, the linear model
comes second, and the quadratic, the tightest in-sample fit, does worst out of sample. Its
curvature, so useful for describing history, makes it unstable the moment it has to extrapolate.

The honest reading is that the warming is well described either as a smooth acceleration or as a
change of pace around 1979, with the data leaning only slightly toward the second. What matters
is what all three agree on: the rate of warming today is several times what it was early in the
twentieth century.

---

## The dashboard

`app.py` is an interactive Plotly Dash app, built as an editorial scroll:

- The raw monthly series and the 1, 5 and 10-year rolling means.
- The three fitted models over the historical record, alongside the cross-validation table.
- Forecasts to 2045 for all three models, with the quadratic drawn dotted and labelled as an
  unreliable extrapolation, and a dropdown to isolate any model.

---

## Running it locally

```bash
# 1. Clone the repository
git clone https://github.com/momopj/Temp-Anomaly-Analysis.git
cd Temp-Anomaly-Analysis

# 2. (optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build the derived data, then run the dashboard
python build_data.py
python app.py
```

The terminal prints a local URL (by default `http://127.0.0.1:8050`); open it in your browser.
Debug mode is off by default; set `DASH_DEBUG=true` while developing.

---

## Project structure

| Path | What it is |
|------|------------|
| `build_data.py` | The data pipeline. Reads the raw file, writes every derived file. |
| `Analysis.ipynb` | The exploratory analysis and modelling, narrated. |
| `app.py` | The Dash dashboard. Read-only. |
| `assets/styles.css` | Styling for the dashboard. |
| `data/` | The raw input plus all derived files. |
| `test_build_data.py` | Plumbing tests for the pipeline. |

---

## Future work

A few directions I would like to take this further:

- Model the **monthly** series rather than only the yearly means, which would bring seasonality
  back into the problem and make it a richer forecasting task.
- Add a classical time-series model such as **ARIMA**, and let it compete on the same held-out
  scores rather than just adding another line.
- Show **uncertainty bands** on the forecasts, so the figures carry a sense of how much they can
  be trusted as they reach further out.

I've learnt a lot building this, and found the data genuinely interesting to work with. There is
plenty more I want to do with it.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
