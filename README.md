# Extreme Rainfall Risk API (Dehradun)

FastAPI backend around your pickled scikit-learn model. Every few hours it downloads the day's
hourly weather from Open-Meteo, builds the model's 36 daily features, scores **next-day extreme
rainfall (>= 50 mm/day)**, stores the result, and optionally sends an alert.

> **This is an extreme-rainfall screening prototype, not a verified cloudburst detector.**
> Validation precision was ~0.29 (about 7 in 10 alerts are false alarms) and the metrics come from
> an earlier split. Word your UI accordingly ("elevated extreme-rainfall risk", never "cloudburst
> will occur"). The score is *not* a calibrated probability (class-balanced logistic regression).

## Architecture

```
Open-Meteo (hourly) -> weather.py -> features.py (36 daily features) -> model_service.py
                                                                          |
   scheduler.py (cron) / POST /predictions/run  ->  service.py  ->  storage.py (SQLite) -> alerts.py
                                                                          |
                                   FastAPI (main.py): /predictions/latest, /predictions, /health ...
```

| File | Role |
|---|---|
| `app/features.py` | Parses each feature name into a rule (`*_mean/min/max/sum`, `hours_ge_*mm`, calendar) |
| `app/weather.py` | Open-Meteo client: retries, pinned units, forecast-vs-archive selection |
| `app/model_service.py` | Loads the Pipeline, validates input, scores, explains top factors |
| `app/service.py` | Orchestration, one-alert-per-target-date logic, job bookkeeping |
| `app/storage.py` | SQLite: predictions, observed outcomes, alerts, job runs |
| `app/main.py` | HTTP API, CORS, admin-key auth, scheduler lifecycle |

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # pinned versions, see "scikit-learn pin" below
cp .env.example .env                         # set ADMIN_API_KEY
python -m unittest discover -s tests -t . -v # tests
uvicorn app.main:create_app --factory --reload
# docs: http://localhost:8000/docs
```

Docker: `cp .env.example .env && docker compose up --build -d`

**Use `--factory` and a single worker.** The scheduler runs inside the process; multiple workers
would each run it (duplicate runs and alerts).

## API

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /predictions/latest` | – | Latest stored prediction + `age_minutes`, `stale` (what the frontend polls) |
| `GET /predictions?limit=&start=&end=` | – | History, latest run per target date |
| `POST /predict` | – | Score a manual feature set (not stored). Missing values allowed up to 25% |
| `GET /model-info` | – | Features, threshold, training/validation info, caveats |
| `GET /health` | – | DB, scheduler, last job run, latest prediction age |
| `GET /metrics/live` | – | Precision/recall vs. recorded outcomes |
| `POST /predictions/run` | `X-API-Key` | Run the pipeline now (`feature_date`, `source`, `persist`, `notify`) |
| `GET /features/preview` | `X-API-Key` | Show computed features for a date (use to compare with training data) |
| `PUT /outcomes/{target_date}` | `X-API-Key` | Record observed rainfall (mm) so live metrics work |

`feature_date` is the day whose weather is the input; the prediction is for the **following** day.

Example `/predictions/latest` (from synthetic input):
```json
{
  "predicted_probability": 0.6544, "threshold": 0.875, "prediction": 0,
  "risk_label": "Below model alert threshold", "target": "Next-day extreme rainfall",
  "feature_date": "2026-07-15", "target_date": "2026-07-16", "location": "Dehradun",
  "data_source": "forecast", "missing_features": [],
  "top_factors": [{"feature": "dew_point_2m_mean", "value": 22.0, "contribution": 2.174, "effect": "raises risk score"}],
  "warnings": ["'year' (2026) is outside the training period (ends 2024); the model extrapolates its year trend."],
  "age_minutes": 12.4, "stale": false
}
```

## Things you MUST verify before trusting predictions

These are assumptions I could not confirm from the files you gave me. Each one changes the
inputs the model sees, silently. Check them against your training notebook; use
`GET /features/preview?feature_date=YYYY-MM-DD&source=archive` for a date in your training set and
compare with your training row.

1. **scikit-learn pin.** The pickle was trained with **1.4.2**. Under a newer version (I tested 1.8)
   it crashes at prediction time. `requirements.txt` pins 1.4.2 and the app refuses to start on a
   mismatch. Longer term: re-export the model with a current version.
2. **Day boundary / timezone.** Default `Asia/Kolkata`. If training used GMT days, set `WEATHER_TIMEZONE=GMT`.
3. **`hours_ge_5mm` / `hours_ge_10mm`** are computed from hourly `precipitation` (not `rain`).
4. **`wind_direction_10m_mean`** is a plain arithmetic mean of hourly directions.
5. **Units** are Open-Meteo defaults (°C, km/h, hPa, mm). Training medians are consistent with these.
6. **Training/serving skew.** The training history (1990-2024) was almost certainly ERA5 reanalysis;
   live input comes from forecast-model data, which differs somewhat. Measure it (see 8).
7. **`year` is a model feature** and live dates (2026+) are beyond the training range (ends 2024).
   Every response carries a warning. Consider re-training without `year`.
8. **Measure real performance.** Record observed rainfall with `PUT /outcomes/{date}`; `/metrics/live`
   then reports precision/recall. Extreme days are rare, so expect to wait for many events.

Location: coordinates default to central Dehradun. Your labels use the max over four grid cells;
the weather inputs here are for one point. Change `LATITUDE/LONGITUDE` if training used another point.

## Operations

- **Failure behaviour:** if Open-Meteo is down, the job logs and records the failure
  (`/health` -> `last_run`); `/predictions/latest` keeps serving the last result with `stale: true`
  after `STALE_AFTER_HOURS`. The frontend should show a "data is old" banner when `stale` is true.
- **Alerts:** at most one per target date, only if delivery succeeded (failed webhooks retry next run).
  `ALERT_WEBHOOK_URL` takes a Slack/Discord/custom webhook. For SMS/email/push, extend `app/alerts.py`.
- **Security:** admin endpoints are disabled unless `ADMIN_API_KEY` is set. Restrict `CORS_ORIGINS` to
  your frontend. Put the API behind HTTPS (reverse proxy/platform). Add rate limiting at the proxy.
- **Open-Meteo terms:** the free API is for non-commercial use; a commercial deployment needs a paid plan.
- **Scaling later:** swap SQLite for PostgreSQL by reimplementing `Store`; move the scheduler to a
  separate worker/cron if you need several API replicas.
