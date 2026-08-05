"""LSTM ve GRU tabanlı fiyat tahmin modelleri.

Girdi: (batch, lookback, 1) ölçeklenmiş kapanış fiyatı penceresi.
Çıktı: (batch, 1) bir sonraki günün tahmini (ölçekli).
"""
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_SIZE = 32
NUM_LAYERS = 2


class StockLSTM(nn.Module):
    """input_size=1 -> LSTM(hidden=32, layers=2) -> son zaman adımı -> Linear(32, 1)."""

    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))   # out: (batch, lookback, hidden_size)
        last_step = out[:, -1, :]                # (batch, hidden_size) -> son zaman adımı
        return self.fc(last_step)                # (batch, 1)


class StockGRU(nn.Module):
    """StockLSTM ile birebir aynı yapı; tek fark nn.GRU ve cell state olmaması (sadece h0)."""

    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, hn = self.gru(x, h0)                # out: (batch, lookback, hidden_size)
        last_step = out[:, -1, :]                # (batch, hidden_size) -> son zaman adımı
        return self.fc(last_step)                # (batch, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, lookback, input_size = 4, 20, 1
    x = torch.randn(batch, lookback, input_size)

    for name, model in [("StockLSTM", StockLSTM()), ("StockGRU", StockGRU())]:
        y = model(x)
        print(f"[models] {name} input  shape: {tuple(x.shape)}")
        print(f"[models] {name} output shape: {tuple(y.shape)}")
