"""Karşılaştırma için iki baseline: Naive (dünkü fiyat) ve Linear Regression.

Tüm metrikler dolar biriminde (ölçeklenmemiş gerçek fiyatlar üzerinde) hesaplanır.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"


def _rmse(y_true, y_pred) -> float:
    """Dolar biriminde RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def naive_baseline(test_prices: np.ndarray):
    """Naive tahmin: tahmin[t] = gerçek[t-1] (dünkü fiyatı yarına kopyalar).

    test_prices: (n,) dolar cinsinden gerçek test fiyatları.
    (y_true, y_pred) döndürür; ilk gün tahmin edilemez, atlanır.
    """
    prices = np.asarray(test_prices, dtype=np.float64).reshape(-1)
    y_true = prices[1:]        # t = 1..n-1 gerçek değerler
    y_pred = prices[:-1]       # t-1 kaydırılmış (dünkü fiyat) tahmin
    return y_true, y_pred


def linear_regression_baseline(train_prices: np.ndarray, test_prices: np.ndarray,
                               lookback: int = 20):
    """Düzleştirilmiş 20 günlük pencere (n, 20) ile Linear Regression baseline'ı.

    Dolar cinsinden ham fiyatlarla çalışır (ölçekleme baseline için gerekmez).
    Train penceresiyle fit eder, test penceresinde tahmin üretir.
    (y_true, y_pred) döndürür.
    """
    def windows(prices):
        prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        X, y = [], []
        for i in range(len(prices) - lookback):
            X.append(prices[i:i + lookback])   # (lookback,) düz pencere
            y.append(prices[i + lookback])     # takip eden gün
        return np.array(X), np.array(y)        # X: (m, lookback), y: (m,)

    X_train, y_train = windows(train_prices)
    X_test, y_test = windows(test_prices)

    model = LinearRegression()
    model.fit(X_train, y_train)          # girdi zaten (m, 20) düz -> reshape gerekmez
    y_pred = model.predict(X_test)
    return y_test, y_pred


def compute_baselines(train_prices: np.ndarray, test_prices: np.ndarray,
                      lookback: int = 20, save: bool = True) -> dict:
    """İki baseline'ın dolar RMSE/MAE değerlerini hesaplar ve JSON'a yazar."""
    # Naive
    yt_n, yp_n = naive_baseline(test_prices)
    naive = {"rmse": _rmse(yt_n, yp_n), "mae": float(mean_absolute_error(yt_n, yp_n))}

    # Linear Regression
    yt_l, yp_l = linear_regression_baseline(train_prices, test_prices, lookback)
    linreg = {"rmse": _rmse(yt_l, yp_l), "mae": float(mean_absolute_error(yt_l, yp_l))}

    metrics = {"naive": naive, "linear_regression": linreg}

    print(f"[baselines] Naive             -> RMSE: ${naive['rmse']:.2f} | MAE: ${naive['mae']:.2f}")
    print(f"[baselines] LinearRegression  -> RMSE: ${linreg['rmse']:.2f} | MAE: ${linreg['mae']:.2f}")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[baselines] Metrikler kaydedildi: {METRICS_PATH.name}")

    return metrics


if __name__ == "__main__":
    from data_loader import load_stock_data
    from explore import explore
    from preprocessing import chronological_split

    df = explore(load_stock_data())
    tr, te = chronological_split(df["Close"])
    compute_baselines(tr.values, te.values)
