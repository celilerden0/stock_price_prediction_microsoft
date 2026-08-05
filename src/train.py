"""Eğitim döngüsü: MSE + Adam ile mini-batch eğitim.

shuffle=True gerekçesi: create_sequences ile pencereler oluşturulduktan sonra
her (X, y) örneği bağımsızdır ve model örnekler arasında durum taşımıyor
(StockLSTM.forward her çağrıda h0/c0'ı sıfırdan başlatıyor). Kronolojik kısıt
zaten chronological_split aşamasında sağlandı (train/test asla karışmaz);
train setinin kendi içindeki örnek sırası eğitim sırasında karıştırılabilir,
bu bir sızıntı yaratmaz, sadece SGD'nin batch'ler arası korelasyonunu kırar.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int = 100, lr: float = 0.01, batch_size: int = 32):
    """Verilen modeli MSE + Adam ile eğitir.

    Her 10 epoch'ta ortalama loss yazdırır.
    (egitilmis model, epoch basina ortalama loss listesi, toplam sure saniye) döndürür.
    """
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)  # batch ortalamasini toplama geri cevir

        epoch_loss = running_loss / len(dataset)  # bu epoch'un ornek basina ortalama loss'u
        epoch_losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"[train] epoch {epoch:>4}/{epochs} | loss: {epoch_loss:.6f}")

    elapsed = time.perf_counter() - start
    print(f"[train] Egitim tamamlandi: {elapsed:.2f}s ({epochs} epoch)")
    return model, epoch_losses, elapsed


if __name__ == "__main__":
    from data_loader import load_stock_data
    from explore import explore
    from models import StockLSTM
    from preprocessing import chronological_split, fit_scaler, transform_series
    from windowing import make_tensors

    df = explore(load_stock_data())
    tr, te = chronological_split(df["Close"])
    sc = fit_scaler(tr)
    Xtr, ytr = make_tensors(transform_series(sc, tr), name="train")

    model = StockLSTM()
    _, losses, _ = train_model(model, Xtr, ytr, epochs=10)

    assert losses[-1] < losses[0], "10 epoch sonunda loss dusmedi!"
    print(f"[train] Dogrulama OK: loss {losses[0]:.6f} -> {losses[-1]:.6f}")
