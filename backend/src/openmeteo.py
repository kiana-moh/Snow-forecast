from __future__ import annotations
import requests
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_gusts_10m",
]

def fetch_hourly_archive(lat: float, lon: float, start_date: str, end_date: str, tz: str = "America/Toronto") -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": tz,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    hourly = j["hourly"]
    df = pd.DataFrame({"time": hourly["time"]})
    for v in HOURLY_VARS:
        df[v] = hourly.get(v)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df
