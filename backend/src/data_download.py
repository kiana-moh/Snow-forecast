from __future__ import annotations
import os
from tqdm import tqdm
from resorts import RESORTS
from openmeteo import fetch_hourly_archive
from utils import ensure_dir

RAW_DIR = "data/raw"

def main():
    ensure_dir(RAW_DIR)

    # Pick a training range (2–3 years is enough)
    start_date = "2022-01-01"
    end_date = "2025-12-31"

    for r in tqdm(RESORTS, desc="Downloading"):
        df = fetch_hourly_archive(r["lat"], r["lon"], start_date, end_date)
        out = os.path.join(RAW_DIR, f"{r['id']}.csv")
        df.to_csv(out, index=False)

if __name__ == "__main__":
    main()
