import numpy as np

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ski_score_from_next24h(
    new_snow_cm: float,
    rain_mm: float,
    temp_max_c: float,
    temp_min_c: float,
    wind_gust_max_kmh: float,
    snow_depth_m: float | None,
) -> float:
    # Base
    score = 50.0

    # Fresh snow bonus (0–30 points, saturates at 20cm)
    score += 30.0 * clamp(new_snow_cm / 20.0, 0.0, 1.0)

    # Rain is brutal
    if rain_mm >= 5: score -= 45
    elif rain_mm >= 1: score -= 30
    elif rain_mm > 0: score -= 15

    # Temperature penalties (slush/ice risk)
    if temp_max_c >= 3: score -= 20
    elif temp_max_c >= 1: score -= 10
    if temp_min_c <= -20: score -= 10  # very cold can feel icy/brittle

    # Wind / lift-hold risk
    if wind_gust_max_kmh >= 70: score -= 20
    elif wind_gust_max_kmh >= 50: score -= 10

    # Base depth (small bonus if we have it)
    if snow_depth_m is not None:
        if snow_depth_m >= 1.0: score += 5
        elif snow_depth_m <= 0.3: score -= 5

    return float(clamp(score, 0.0, 100.0))
