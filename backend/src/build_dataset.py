import numpy as np
import pandas as pd
from score import ski_score_from_next24h

FEATURES = [
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_gusts_10m",
]

def make_windows(df: pd.DataFrame, window_hours: int = 72):
    df = df.dropna(subset=["temperature_2m"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    X, y, t = [], [], []

    vals = df[FEATURES].to_numpy(dtype=np.float32)

    # Precompute next-24h aggregates for labels
    snowfall = df["snowfall"].to_numpy(np.float32)          # cm per hour (Open-Meteo provides snowfall hourly) :contentReference[oaicite:2]{index=2}
    rain = df["rain"].to_numpy(np.float32)                  # mm per hour
    temp = df["temperature_2m"].to_numpy(np.float32)
    gust = df["wind_gusts_10m"].to_numpy(np.float32)
    depth = df["snow_depth"].to_numpy(np.float32)

    n = len(df)
    for i in range(window_hours, n - 24):
        past = vals[i - window_hours:i]           # [W, F]
        # next 24 hours (label window)
        j0, j1 = i, i + 24
        new_snow_cm = float(np.nansum(snowfall[j0:j1]))
        rain_mm = float(np.nansum(rain[j0:j1]))
        temp_max = float(np.nanmax(temp[j0:j1]))
        temp_min = float(np.nanmin(temp[j0:j1]))
        gust_max = float(np.nanmax(gust[j0:j1]))
        depth_mean = float(np.nanmean(depth[j0:j1])) if np.isfinite(depth[j0:j1]).any() else None

        score = ski_score_from_next24h(new_snow_cm, rain_mm, temp_max, temp_min, gust_max, depth_mean)

        X.append(past)
        y.append(score)
        t.append(df.loc[i, "time"])

    return np.stack(X), np.array(y, dtype=np.float32), pd.to_datetime(t)
