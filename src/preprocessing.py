"""Kronolojik train/test bölme ve veri sızıntısız (leakage-free) ölçekleme.

Kritik kısıt: MinMaxScaler SADECE train'e fit edilir; test'e yalnızca transform uygulanır.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"


def chronological_split(series: pd.Series, train_ratio: float = 0.8):
    """Seriyi kronolojik olarak böler: ilk %80 train, son %20 test. Shuffle YOK.

    Zaman serisi sırası korunur (geleceği görmemek için karıştırma yapılmaz).
    (train_series, test_series) döndürür.
    """
    n = len(series)
    split_idx = int(n * train_ratio)  # Bölme noktası: ilk %80'in bittiği indeks.
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    print(f"[preprocessing] Toplam: {n} | Train: {len(train)} | Test: {len(test)}")
    print(f"[preprocessing] Train tarih: {train.index.min().date()} -> {train.index.max().date()}")
    print(f"[preprocessing] Test  tarih: {test.index.min().date()} -> {test.index.max().date()}")
    return train, test


def fit_scaler(train: pd.Series, save: bool = True) -> MinMaxScaler:
    """MinMaxScaler(feature_range=(-1, 1))'i SADECE train verisine fit eder.

    Scaler'ı results/scaler.pkl olarak kaydeder (commit'lenecek). Fit edilmiş scaler döndürür.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # (n,) -> (n, 1): sklearn 2B girdi bekler. Sadece train ile fit -> leakage yok.
    scaler.fit(train.values.reshape(-1, 1))

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[preprocessing] Scaler kaydedildi: {SCALER_PATH.name} "
              f"(fit araligi: {scaler.data_min_[0]:.2f} - {scaler.data_max_[0]:.2f})")
    return scaler


def transform_series(scaler: MinMaxScaler, series: pd.Series) -> np.ndarray:
    """Bir seriyi fit edilmiş scaler ile ölçekler. (n, 1) float64 numpy döndürür."""
    # (n,) -> (n, 1) transform sonrası da (n, 1) kalır.
    return scaler.transform(series.values.reshape(-1, 1))


if __name__ == "__main__":
    from data_loader import load_stock_data
    from explore import explore

    df = explore(load_stock_data())
    tr, te = chronological_split(df["Close"])
    sc = fit_scaler(tr)
    print("train scaled shape:", transform_series(sc, tr).shape)
