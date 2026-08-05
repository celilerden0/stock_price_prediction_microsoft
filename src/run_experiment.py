"""LSTM vs GRU deneyi: aynı veri, aynı hiperparametreler, 100 epoch.

Kısıt: Model, ham fiyat SEVİYESİ yerine gunluk log-return tahmin edecek şekilde
eğitilir (bkz. preprocessing.to_log_returns). Fiyat seviyesiyle MinMaxScaler fit
edilirse test döneminde fiyat train aralığının (max $186) dışına çıktığında
(test'te $284'e kadar çıkıyor) model ekstrapolasyon yapmak zorunda kalıp
sistematik olarak düşük tahmin üretiyordu. Return'ler dar ve durağan bir aralıkta
kaldığı için bu sorunu ortadan kaldırır.

Kısıt: Adil karşılaştırma için seed her modelden ÖNCE 42'ye resetlenir
(model ağırlık başlatma ve DataLoader shuffle sırası ikisinde de aynı olsun diye).

Çıktılar (results/):
- lstm_model.pt, gru_model.pt      -> egitilmis state_dict'ler
- loss_history.json                 -> epoch basina loss (her iki model)
- training_times.json               -> toplam egitim suresi (saniye)
- loss_curves.png                   -> iki loss egrisi tek grafikte
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Ekransız (backend) kayıt için.
import matplotlib.pyplot as plt
import torch

from data_loader import load_stock_data
from explore import explore
from models import StockGRU, StockLSTM
from preprocessing import chronological_split, fit_scaler, to_log_returns, transform_series
from train import train_model
from windowing import make_tensors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42
EPOCHS = 100


def run_experiment():
    """LSTM ve GRU'yu aynı train verisiyle eğitir, sonuçları results/ altına kaydeder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = explore(load_stock_data())
    tr, te = chronological_split(df["Close"])
    tr_returns = to_log_returns(tr)   # train icin "onceki gun" yok -> ilk gun return=0
    sc = fit_scaler(tr_returns)       # scaler artik fiyat degil, RETURN uzerine fit ediliyor
    Xtr, ytr = make_tensors(transform_series(sc, tr_returns), name="train")

    loss_history = {}
    training_times = {}

    experiments = [("lstm", StockLSTM), ("gru", StockGRU)]
    for name, model_cls in experiments:
        torch.manual_seed(SEED)  # her model ayni baslangic kosullariyla basliyor
        model = model_cls()

        print(f"[run_experiment] --- {name.upper()} egitimi basliyor ---")
        trained_model, losses, elapsed = train_model(model, Xtr, ytr, epochs=EPOCHS)

        loss_history[name] = losses
        training_times[name] = elapsed
        torch.save(trained_model.state_dict(), RESULTS_DIR / f"{name}_model.pt")
        print(f"[run_experiment] {name}_model.pt kaydedildi.")

    with open(RESULTS_DIR / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)
    print(f"[run_experiment] loss_history.json kaydedildi.")

    with open(RESULTS_DIR / "training_times.json", "w", encoding="utf-8") as f:
        json.dump(training_times, f, indent=2)
    print(f"[run_experiment] training_times.json kaydedildi.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history["lstm"], label="LSTM", color="tab:blue")
    ax.plot(loss_history["gru"], label="GRU", color="tab:orange")
    ax.set_title(f"LSTM vs GRU - Egitim Loss ({EPOCHS} epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = RESULTS_DIR / "loss_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[run_experiment] loss_curves.png kaydedildi.")

    print(f"[run_experiment] Sureler -> LSTM: {training_times['lstm']:.2f}s | "
          f"GRU: {training_times['gru']:.2f}s")
    return loss_history, training_times


if __name__ == "__main__":
    run_experiment()
