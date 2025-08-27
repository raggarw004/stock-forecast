#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- utils ---------------------- #

def to_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1)

def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = to_1d(y_true), to_1d(y_pred)
    true_dir = np.sign(np.diff(yt, prepend=yt[0]))
    pred_dir = np.sign(np.diff(yp, prepend=yp[0]))
    n = min(len(true_dir), len(pred_dir))
    return float((true_dir[:n] == pred_dir[:n]).mean()) if n else 0.0

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt, yp = to_1d(y_true), to_1d(y_pred)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-8, None))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "DirAcc%": 100.0 * direction_accuracy(yt, yp)}

# ---------------------- data ---------------------- #

def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = df[["Close"]].dropna().copy()
    df.index.name = "Date"
    return df

def make_features(df: pd.DataFrame, lags=(1,2,3,5,10), windows=(5,10,20)) -> pd.DataFrame:
    X = df.copy()
    X["Return"] = X["Close"].pct_change()
    for l in lags:
        X[f"Lag_{l}"] = X["Close"].shift(l)
        X[f"RetLag_{l}"] = X["Return"].shift(l)
    for w in windows:
        X[f"RollMean_{w}"] = X["Close"].rolling(w).mean()
        X[f"RollStd_{w}"]  = X["Close"].rolling(w).std()
        X[f"RollRet_{w}"]  = X["Return"].rolling(w).mean()
    X = X.dropna()
    return X

def time_split_index(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.Index, pd.Index]:
    n = len(df)
    k = max(1, int(n * (1 - test_size)))
    return df.index[:k], df.index[k:]

# ---------------------- ARIMA ---------------------- #

@dataclass
class ARIMAConfig:
    order: Tuple[int,int,int] = (5,1,0)

def run_arima(close: pd.Series, test_idx: pd.Index, cfg: ARIMAConfig):
    train_series = close.loc[: test_idx[0]].iloc[:-1]
    model = ARIMA(train_series, order=cfg.order)
    fitted = model.fit()
    steps = len(test_idx)
    fc = fitted.forecast(steps=steps)
    fc.index = test_idx
    return to_1d(close.loc[test_idx].values), to_1d(fc.values)

# ---------------------- Gradient Boosting ---------------------- #

def run_gbr(X: pd.DataFrame, target_col: str, test_idx: pd.Index):
    y = X[target_col].copy()
    feats = X.drop(columns=[target_col])

    X_train = feats.loc[: test_idx[0]].iloc[:-1]
    y_train = y.loc[X_train.index]
    X_test  = feats.loc[test_idx]
    y_test  = y.loc[test_idx]

    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=42
    )
    model.fit(X_train.values, to_1d(y_train.values))
    y_pred = model.predict(X_test.values)
    return to_1d(y_test.values), to_1d(y_pred)

# ---------------------- LSTM ---------------------- #

class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, lookback: int):
        self.x, self.y = [], []
        for i in range(lookback, len(series)):
            self.x.append(series[i-lookback:i])
            self.y.append(series[i])
        self.x = np.stack(self.x)  # (N, L)
        self.y = np.array(self.y)  # (N,)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: (B, L, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (B,)

@dataclass
class LSTMConfig:
    lookback: int = 60
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    val_frac: float = 0.1
    early_patience: int = 5

def run_lstm(close: pd.Series, test_idx: pd.Index, cfg: LSTMConfig):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # scale on train only
    train_series = close.loc[: test_idx[0]].iloc[:-1].values.astype(float)
    test_series  = close.loc[test_idx].values.astype(float)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1,1)).reshape(-1)
    test_scaled  = scaler.transform(test_series.reshape(-1,1)).reshape(-1)

    # build sequence datasets
    ds_train = SeqDataset(train_scaled, cfg.lookback)

    # split train/val
    n = len(ds_train)
    n_val = max(1, int(n * cfg.val_frac))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = LSTMRegressor(input_dim=1, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = cfg.early_patience
    for epoch in range(1, cfg.epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.unsqueeze(-1).to(device)   # (B, L, 1)
            yb = yb.to(device)                 # (B,)
            opt.zero_grad()
            pred = model(xb)                   # (B,)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(yb)
        train_loss /= max(1, len(train_loader.dataset))

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.unsqueeze(-1).to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(yb)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_patience
        else:
            patience -= 1
        if patience == 0:
            break

    # restore best
    model.load_state_dict(best_state)

    # walk-forward on test set (one-step ahead)
    # start buffer: last lookback points from train
    hist = train_scaled[-cfg.lookback:].tolist()
    preds_scaled: List[float] = []
    model.eval()
    with torch.no_grad():
        for t in range(len(test_scaled)):
            x = torch.tensor(hist[-cfg.lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            p = model(x).item()
            preds_scaled.append(p)
            # teacher forcing w/ true next value (classical 1-step ahead evaluation)
            hist.append(test_scaled[t])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).reshape(-1)
    y_true = close.loc[test_idx].values.astype(float)
    return to_1d(y_true), to_1d(preds)

# ---------------------- plotting ---------------------- #

def plot_results(dates, close, preds: Dict[str, np.ndarray], out_path=None):
    plt.figure(figsize=(11, 5.5))
    plt.plot(close.index, close.values, label="Actual Close")
    for name, arr in preds.items():
        plt.plot(dates, arr, label=name)
    plt.title("Stock Price: Actual vs Forecasts")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=144)
    plt.show()

# ---------------------- main ---------------------- #

def main():
    ap = argparse.ArgumentParser(description="ARIMA vs GBR vs LSTM forecasting")
    ap.add_argument("--ticker", type=str, default="AAPL")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--arima", type=str, default="5,1,0")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save_plot", type=str, default="")
    args = ap.parse_args()

    order = tuple(int(x) for x in args.arima.split(","))
    ar_cfg = ARIMAConfig(order=order)
    lstm_cfg = LSTMConfig(lookback=args.lookback, epochs=args.epochs)

    print(f"Downloading {args.ticker}…")
    df = fetch_data(args.ticker, args.years)
    X = make_features(df)
    train_idx, test_idx = time_split_index(X, test_size=args.test_size)
    print(f"Samples: total={len(X)}, train={len(train_idx)}, test={len(test_idx)}")

    # ARIMA
    y_true_a, y_pred_a = run_arima(df["Close"], test_idx, ar_cfg)
    m_arima = metrics(y_true_a, y_pred_a)

    # GBR
    y_true_g, y_pred_g = run_gbr(X, "Close", test_idx)
    m_gbr = metrics(y_true_g, y_pred_g)

    # LSTM
    y_true_l, y_pred_l = run_lstm(df["Close"], test_idx, lstm_cfg)
    m_lstm = metrics(y_true_l, y_pred_l)

    def fmt(m): return ", ".join(f"{k}={v:.4f}" if not str(k).endswith("%") else f"{k}={v:.2f}" for k, v in m.items())
    print("\n=== Metrics (Test) ===")
    print(f"ARIMA{ar_cfg.order}: {fmt(m_arima)}")
    print(f"GBR:               {fmt(m_gbr)}")
    print(f"LSTM(L={lstm_cfg.lookback}): {fmt(m_lstm)}")

    preds = {"ARIMA": y_pred_a, "GBR": y_pred_g, "LSTM": y_pred_l}
    plot_results(test_idx, df["Close"], preds, out_path=(args.save_plot or None))

if __name__ == "__main__":
    main()
