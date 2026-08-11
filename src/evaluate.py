"""Naive, Linear Regression, LSTM ve GRU'yu aynı test gün kümesinde karşılaştırır.

Kritik kısıt: baseline_metrics.json'daki Naive metriği FARKLI bir gün kümesinde
hesaplanmış (test[1:], 833 gün) — LSTM/GRU/LinReg ise pencereleme (lookback=20)
yüzünden test[20:] üzerinde çalışıyor (814 gün). Bu ikisini doğrudan karşılaştırmak
yanıltıcı olurdu, bu yüzden Naive burada ortak 814 günlük kümeye kısıtlanarak
YENİDEN hesaplanır. Linear Regression zaten aynı pencerelemeyle üretildiği için
baseline_metrics.json'daki değeri doğrudan (ve doğrulanarak) kullanılır.

Kritik kısıt: LSTM/GRU artık log-return tahmin ediyor (run_experiment.py), fiyat
değil. Dolar cinsinden karşılaştırma için tahmin edilen return, bir önceki günün
GERÇEK fiyatına uygulanarak (price_pred[t] = price_true[t-1] * exp(return_pred[t]))
tek-adım (one-step) dolar fiyatına geri dönüştürülür. Bu, Naive/LinReg'in de her
zaman gerçek geçmişten tahmin ürettiği yaklaşımla tutarlıdır (hata biriktirmez).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baselines import linear_regression_baseline, naive_baseline
from data_loader import load_stock_data
from explore import explore
from models import StockGRU, StockLSTM
from preprocessing import SCALER_PATH, chronological_split, to_log_returns, transform_series
from windowing import LOOKBACK, make_tensors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true, y_pred) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _load_model(cls, name: str):
    model = cls()
    model.load_state_dict(torch.load(RESULTS_DIR / f"{name}_model.pt"))
    model.eval()
    return model


def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = explore(load_stock_data())
    tr, te = chronological_split(df["Close"])
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)  # egitimde kullanilan, RETURN scaler'i (yeniden fit YOK)

    # test'in ilk gunu icin "onceki gun" olarak train'in son GERCEK fiyati kullanilir
    # (gelecek bilgisi degil, sadece kronolojik surekliligi korur -- bkz. to_log_returns).
    te_returns = to_log_returns(te, prev_price=tr.values[-1])
    Xte, yte = make_tensors(transform_series(scaler, te_returns), name="test")

    # --- LSTM / GRU: scaled return -> gercek return -> dolar fiyati ---
    lstm = _load_model(StockLSTM, "lstm")
    gru = _load_model(StockGRU, "gru")
    with torch.no_grad():
        lstm_ret_scaled = lstm(Xte).numpy()
        gru_ret_scaled = gru(Xte).numpy()

    lstm_ret = scaler.inverse_transform(lstm_ret_scaled).reshape(-1)
    gru_ret = scaler.inverse_transform(gru_ret_scaled).reshape(-1)

    # hedef gun i+LOOKBACK icin "bir onceki gun"un GERCEK fiyati: test[LOOKBACK-1 : -1]
    prev_actual_price = te.values[LOOKBACK - 1: LOOKBACK - 1 + len(lstm_ret)]
    lstm_pred = prev_actual_price * np.exp(lstm_ret)
    gru_pred = prev_actual_price * np.exp(gru_ret)
    dl_actual = te.values[LOOKBACK:LOOKBACK + len(lstm_ret)]        # (814,) ortak gun kumesi, dogrudan gercek fiyat

    # --- Linear Regression: zaten ayni pencerelemeyle (lookback=20) uretiliyor ---
    lr_true, lr_pred = linear_regression_baseline(tr.values, te.values, lookback=LOOKBACK)

    # --- Naive: tam seri uzerinde hesaplanip ortak gun kumesine (test[LOOKBACK:]) kisitlanir ---
    naive_true_full, naive_pred_full = naive_baseline(te.values)   # test[1:] -> 833 gun
    naive_true = naive_true_full[LOOKBACK - 1:]                    # test[LOOKBACK:] -> 814 gun
    naive_pred = naive_pred_full[LOOKBACK - 1:]

    # --- Ayni gun kumesi dogrulamasi ---
    assert len(dl_actual) == len(lr_true) == len(naive_true), (
        f"Gun kumesi boyutlari uyusmuyor: LSTM/GRU={len(dl_actual)}, "
        f"LinReg={len(lr_true)}, Naive={len(naive_true)}"
    )
    assert np.allclose(dl_actual, lr_true, atol=1e-2), "LinReg gun kumesi LSTM/GRU ile uyusmuyor!"
    assert np.allclose(dl_actual, naive_true, atol=1e-2), "Naive gun kumesi LSTM/GRU ile uyusmuyor!"
    print(f"[evaluate] Dogrulama OK: 4 model de ayni {len(dl_actual)} gunluk test kumesinde "
          f"degerlendirildi (test[{LOOKBACK}:], {te.index[LOOKBACK].date()} -> {te.index[-1].date()}).")

    # --- Metrikler ---
    naive_m = _metrics(naive_true, naive_pred)
    lr_m = _metrics(lr_true, lr_pred)
    lstm_m = _metrics(dl_actual, lstm_pred)
    gru_m = _metrics(dl_actual, gru_pred)

    with open(RESULTS_DIR / "training_times.json") as f:
        training_times = json.load(f)

    rows = [
        ("Naive", naive_m, None),
        ("Linear Regression", lr_m, None),
        ("LSTM", lstm_m, training_times["lstm"]),
        ("GRU", gru_m, training_times["gru"]),
    ]

    # --- Tablo (konsol + markdown) ---
    header = f"{'Model':<20} | {'MSE':>12} | {'RMSE ($)':>10} | {'MAE ($)':>10} | {'Egitim suresi (s)':>18}"
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)
    md_lines = ["| Model | MSE | RMSE ($) | MAE ($) | Egitim suresi (s) |",
                "|---|---|---|---|---|"]
    for name, m, t in rows:
        t_str = f"{t:.2f}" if t is not None else "-"
        print(f"{name:<20} | {m['mse']:>12.4f} | {m['rmse']:>10.4f} | {m['mae']:>10.4f} | {t_str:>18}")
        md_lines.append(f"| {name} | {m['mse']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | {t_str} |")
    print()

    with open(RESULTS_DIR / "comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[evaluate] comparison.md kaydedildi.")

    # --- predictions.png: tum test donemi, gercek + 4 tahmin ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(te.index, te.values, label="Gercek", color="black", linewidth=1.2)
    ax.plot(te.index[1:], naive_pred_full, label="Naive", color="tab:gray", linewidth=1, alpha=0.8)
    ax.plot(te.index[LOOKBACK:], lr_pred, label="Linear Regression", color="tab:green", linewidth=1, alpha=0.8)
    ax.plot(te.index[LOOKBACK:], lstm_pred, label="LSTM", color="tab:blue", linewidth=1, alpha=0.9)
    ax.plot(te.index[LOOKBACK:], gru_pred, label="GRU", color="tab:orange", linewidth=1, alpha=0.9)
    ax.set_title("Test Donemi: Gercek Fiyat vs 4 Model Tahmini")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predictions.png", dpi=120)
    plt.close(fig)
    print(f"[evaluate] predictions.png kaydedildi.")

    # --- predictions_zoom.png: son 100 gun, gercek + LSTM + GRU ---
    n_zoom = 100
    zoom_idx = te.index[LOOKBACK:][-n_zoom:]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(zoom_idx, dl_actual[-n_zoom:], label="Gercek", color="black", linewidth=1.5, marker="o", markersize=3)
    ax.plot(zoom_idx, lstm_pred[-n_zoom:], label="LSTM", color="tab:blue", linewidth=1.2, marker="o", markersize=3)
    ax.plot(zoom_idx, gru_pred[-n_zoom:], label="GRU", color="tab:orange", linewidth=1.2, marker="o", markersize=3)
    ax.set_title(f"Son {n_zoom} Gun: Gercek vs LSTM vs GRU")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predictions_zoom.png", dpi=120)
    plt.close(fig)
    print(f"[evaluate] predictions_zoom.png kaydedildi.")

    return rows


if __name__ == "__main__":
    evaluate()