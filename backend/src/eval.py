from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from models import LSTMRegressor, CNN1DRegressor
from utils import mae_rmse, ensure_dir
from baselines import baseline_mean_train
import config

MODEL_DIR = "models"
REPORT_DIR = "reports"

def load_model(kind: str, window: int, n_features: int):
    if kind == "lstm":
        model = LSTMRegressor(n_features=n_features)
    else:
        model = CNN1DRegressor(n_features=n_features)

    path = os.path.join(MODEL_DIR, f"{kind}_window{window}.pt")
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "cnn"], default="lstm")
    parser.add_argument("--window", type=int, default=config.WINDOW_HOURS)
    args = parser.parse_args()

    ensure_dir(REPORT_DIR)

    test_path = os.path.join(REPORT_DIR, f"testset_window{args.window}.npz")
    data = np.load(test_path, allow_pickle=True)
    Xte = data["Xte"]
    yte = data["yte"]
    tte = data["tte"].astype("datetime64[ns]")

    n_features = Xte.shape[-1]
    model = load_model(args.model, args.window, n_features=n_features)

    with torch.no_grad():
        preds = model(torch.tensor(Xte, dtype=torch.float32)).numpy()

    mae, rmse = mae_rmse(yte, preds)
    print(f"{args.model.upper()} TEST | MAE={mae:.3f} | RMSE={rmse:.3f}")

    # Baseline: predict mean of test (simple reference)
    base = np.full_like(yte, yte.mean(), dtype=np.float32)
    b_mae, b_rmse = mae_rmse(yte, base)
    print(f"MEAN BASELINE TEST | MAE={b_mae:.3f} | RMSE={b_rmse:.3f}")

    # Plot: prediction vs actual (first ~500 points)
    k = min(500, len(yte))
    plt.figure()
    plt.plot(tte[:k], yte[:k])
    plt.plot(tte[:k], preds[:k])
    plt.xticks(rotation=45)
    plt.tight_layout()
    out1 = os.path.join(REPORT_DIR, f"{args.model}_pred_vs_actual_window{args.window}.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Saved:", out1)

    # Plot: residual histogram
    resid = (preds - yte).reshape(-1)
    plt.figure()
    plt.hist(resid, bins=40)
    plt.tight_layout()
    out2 = os.path.join(REPORT_DIR, f"{args.model}_residuals_window{args.window}.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Saved:", out2)

if __name__ == "__main__":
    main()
