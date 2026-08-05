"""Ölçeklenmiş 1B seriden kayan pencere (sliding window) örnekleri üretir.

Kritik kısıt: Train ve test AYRI AYRI pencerelenir (önce birleştirip sonra bölmek
sızıntı yaratırdı: bir test penceresi train'e ait günleri içeremez).
"""
from __future__ import annotations

import numpy as np
import torch

LOOKBACK = 20


def create_sequences(data: np.ndarray, lookback: int = LOOKBACK):
    """Ölçeklenmiş 1B seriden (X, y) pencere çiftleri üretir (numpy).

    data: (n, 1) veya (n,) ölçeklenmiş dizi.
    X: (n-lookback, lookback, 1) -> her örnek `lookback` günlük pencere.
    y: (n-lookback, 1)          -> pencereyi takip eden günün değeri (hedef).
    """
    data = np.asarray(data, dtype=np.float32).reshape(-1)  # (n,) düz diziye indir.
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])       # i..i+lookback-1 -> girdi penceresi
        y.append(data[i + lookback])         # i+lookback      -> tahmin edilecek gün
    X = np.array(X, dtype=np.float32)         # (n-lookback, lookback)
    y = np.array(y, dtype=np.float32)         # (n-lookback,)
    X = X[..., np.newaxis]                    # (n-lookback, lookback, 1) -> input_size=1 kanalı
    y = y[..., np.newaxis]                    # (n-lookback, 1)
    return X, y


def make_tensors(data: np.ndarray, lookback: int = LOOKBACK, name: str = ""):
    """create_sequences çıktısını float32 torch tensörüne çevirir ve shape'leri yazdırır."""
    X_np, y_np = create_sequences(data, lookback)
    # torch.from_numpy: numpy belleğini paylaşır; .float() ile float32 garanti.
    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()
    print(f"[windowing] {name} X shape: {tuple(X.shape)} | y shape: {tuple(y.shape)}")
    return X, y


if __name__ == "__main__":
    from data_loader import load_stock_data
    from explore import explore
    from preprocessing import chronological_split, fit_scaler, transform_series

    df = explore(load_stock_data())
    tr, te = chronological_split(df["Close"])
    sc = fit_scaler(tr)
    Xtr, ytr = make_tensors(transform_series(sc, tr), name="train")
    Xte, yte = make_tensors(transform_series(sc, te), name="test")
