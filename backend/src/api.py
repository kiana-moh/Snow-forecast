from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests

from src.resorts import RESORTS
from src.score import ski_score_from_next24h

app = FastAPI(title="Ski Forecast API", version="1.0")

# allow your frontend (localhost:3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "rain", "snowfall", "snow_depth", "wind_gusts_10m"]

def fetch_next24h(lat: float, lon: float, tz="America/Toronto") -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY),
        "forecast_days": 2,
        "timezone": tz,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    h = j["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for v in HOURLY:
        df[v] = h.get(v)
    return df.sort_values("time").reset_index(drop=True)

def score_tomorrow(df: pd.DataFrame):
    local_dates = df["time"].dt.date
    today = local_dates.iloc[0]
    tomorrow = pd.Timestamp(today) + pd.Timedelta(days=1)
    tomorrow_date = tomorrow.date()

    dft = df[local_dates == tomorrow_date]
    if len(dft) < 12:
        dft = df.iloc[:24]
    dft = dft.iloc[:24]

    new_snow_cm = float(np.nansum(dft["snowfall"].to_numpy(np.float32)))
    rain_mm = float(np.nansum(dft["rain"].to_numpy(np.float32)))
    temp_max = float(np.nanmax(dft["temperature_2m"].to_numpy(np.float32)))
    temp_min = float(np.nanmin(dft["temperature_2m"].to_numpy(np.float32)))
    gust_max = float(np.nanmax(dft["wind_gusts_10m"].to_numpy(np.float32)))
    depth_mean = float(np.nanmean(dft["snow_depth"].to_numpy(np.float32))) if dft["snow_depth"].notna().any() else None

    score = ski_score_from_next24h(new_snow_cm, rain_mm, temp_max, temp_min, gust_max, depth_mean)

    return score, {
        "new_snow_cm": new_snow_cm,
        "rain_mm": rain_mm,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "gust_max": gust_max,
        "snow_depth_m": depth_mean,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/rankings/tomorrow")
def rankings_tomorrow():
    rows = []
    for r in RESORTS:
        df = fetch_next24h(r["lat"], r["lon"])
        score, feats = score_tomorrow(df)
        rows.append({
            "id": r["id"],
            "name": r["name"],
            "score": round(score, 1),
            "details": {
                "new_snow_cm": round(float(feats["new_snow_cm"]), 2),
                "rain_mm": round(float(feats["rain_mm"]), 2),
                "temp_min": round(float(feats["temp_min"]), 2),
                "temp_max": round(float(feats["temp_max"]), 2),
                "gust_max": round(float(feats["gust_max"]), 2),
                "snow_depth_m": None if feats["snow_depth_m"] is None else round(float(feats["snow_depth_m"]), 2),
            }
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return {"day": "tomorrow", "resorts": rows}
