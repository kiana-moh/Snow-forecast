from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from build_dataset import make_windows, FEATURES
from models import LSTMRegressor, CNN1DRegressor
from utils import set_seed, ensure_dir, mae_rmse
import config

RAW_DIR = "data/raw"
MODEL_DIR = "models"
REPORT_DIR = "reports"

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def time_split_by_t(X, y, t, train_frac=0.70, val_frac=0.15):
    n = len(t)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return (X[:i_train], y[:i_train], t[:i_train],
            X[i_train:i_val], y[i_train:i_val], t[i_train:i_val],
            X[i_val:], y[i_val:], t[i_val:])

def fit_standardizer(X_train: np.ndarray):
    # X: [N, W, F] -> normalize per feature across all (N*W) rows
    flat = X_train.reshape(-1, X_train.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu, sigma

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma

def build_global_dataset(window_hours: int):
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise RuntimeError("No CSVs found in data/raw. Run: python src/data_download.py")

    X_list, y_list, t_list = [], [], []

    for f in files:
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"])
        # Ensure all needed columns exist
        for col in FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        X, y, t = make_windows(df, window_hours=window_hours)
        X_list.append(X)
        y_list.append(y)
        t_list.append(t)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    t_all = pd.to_datetime(np.concatenate([ti.to_numpy() for ti in t_list]))
    # Sort globally by time (important for time split)
    order = np.argsort(t_all)
    return X_all[order], y_all[order], t_all[order]

def get_model(kind: str, n_features: int):
    if kind == "lstm":
        return LSTMRegressor(n_features=n_features)
    if kind == "cnn":
        return CNN1DRegressor(n_features=n_features)
    raise ValueError("model must be one of: lstm, cnn")

def train_one(model, train_loader, val_loader, device, lr, epochs, patience):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # optimize MSE, report MAE/RMSE
    model.to(device)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        all_y, all_p = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_losses.append(loss_fn(pred, yb).item())
                all_y.append(yb.cpu().numpy())
                all_p.append(pred.cpu().numpy())

        val_loss = float(np.mean(val_losses))
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_p)
        val_mae, val_rmse = mae_rmse(y_true, y_pred)

        print(f"Epoch {epoch:02d} | train_mse={np.mean(tr_losses):.4f} | val_mse={val_loss:.4f} | val_mae={val_mae:.3f} | val_rmse={val_rmse:.3f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "cnn"], default="lstm")
    parser.add_argument("--window", type=int, default=config.WINDOW_HOURS)
    args = parser.parse_args()

    ensure_dir(MODEL_DIR)
    ensure_dir(REPORT_DIR)
    set_seed(config.SEED)

    device = "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print("Device:", device)

    # 1) Build dataset
    X, y, t = build_global_dataset(window_hours=args.window)
    print("Dataset:", X.shape, y.shape)

    # 2) Time split
    Xtr, ytr, ttr, Xva, yva, tva, Xte, yte, tte = time_split_by_t(
        X, y, t, train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC
    )
    print("Split sizes:", len(ytr), len(yva), len(yte))

    # 3) Normalize using TRAIN only
    mu, sigma = fit_standardizer(Xtr)
    Xtr = apply_standardizer(Xtr, mu, sigma)
    Xva = apply_standardizer(Xva, mu, sigma)
    Xte = apply_standardizer(Xte, mu, sigma)

    # Save normalizer
    np.savez(os.path.join(MODEL_DIR, f"scaler_window{args.window}.npz"), mu=mu, sigma=sigma)

    # 4) Loaders
    train_loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowDataset(Xva, yva), batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)

    # 5) Train
    n_features = X.shape[-1]
    model = get_model(args.model, n_features=n_features)
    model = train_one(model, train_loader, val_loader, device, config.LR, config.EPOCHS, config.PATIENCE)

    # Save trained model
    out_path = os.path.join(MODEL_DIR, f"{args.model}_window{args.window}.pt")
    torch.save(model.state_dict(), out_path)
    print("Saved:", out_path)

    # Save test arrays for eval step
    np.savez(os.path.join(REPORT_DIR, f"testset_window{args.window}.npz"),
             Xte=Xte, yte=yte, tte=np.array(tte, dtype="datetime64[ns]"))
    print("Saved testset to reports/")

if __name__ == "__main__":
    main()
