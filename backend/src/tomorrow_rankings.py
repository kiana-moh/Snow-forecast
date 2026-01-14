from __future__ import annotations
import requests
import numpy as np
import pandas as pd
from resorts import RESORTS
from score import ski_score_from_next24h

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY = [
    "temperature_2m",
    "rain",
    "snowfall",
    "snow_depth",
    "wind_gusts_10m",
]

def fetch_next24h(lat: float, lon: float, tz="America/Toronto") -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY),
        "forecast_days": 2,      # get enough hours to cover “tomorrow”
        "timezone": tz,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    h = j["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for v in HOURLY:
        df[v] = h.get(v)
    return df

def score_tomorrow(df: pd.DataFrame):
    # Define “tomorrow” as the next 24 hours starting at the next local midnight
    df = df.sort_values("time").reset_index(drop=True)
    local_dates = df["time"].dt.date
    today = local_dates.iloc[0]
    tomorrow = pd.Timestamp(today) + pd.Timedelta(days=1)
    tomorrow_date = tomorrow.date()

    dft = df[local_dates == tomorrow_date]
    if len(dft) < 12:
        # fallback: use first 24h available
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

def main():
    rows = []
    for r in RESORTS:
        df = fetch_next24h(r["lat"], r["lon"])
        score, feats = score_tomorrow(df)
        rows.append({
            "resort": r["name"],
            "score": round(score, 1),
            **{k: (None if v is None else round(float(v), 2)) for k, v in feats.items()}
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
